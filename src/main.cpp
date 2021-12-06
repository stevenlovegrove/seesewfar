#include <pangolin/gl/gldraw.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/display/widgets.h>
#include <pangolin/display/default_font.h>
#include <pangolin/display/image_view.h>
#include <pangolin/var/var.h>
#include <pangolin/video/video.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/plot/plotter.h>

#include <sophus/se3.hpp>
#include <sophus/geometry.hpp>

#include <async++.h>

#include <unordered_map>
#include <deque>
#include <thread>
#include <future>
#include <algorithm>
#include <queue>
#include <cmath>

#include "image_loading.h"
#include "utils.h"
#include "vignette_histogram.h"
#include "bspline.h"
#include "star_map.h"
#include "coord_convention.h"
#include "peaks.h"

using namespace pangolin;

constexpr double EARTH_ROTATION_RAD_PER_SECOND= M_PI / (12.0*60.0*60.0);

// TODO:
// * auto tracking.
// * Align to star map
// * dynamic resolution based on view window?

int main( int argc, char** argv )
{
//    test_star_map();
//    return 0;

    const std::string shader_dir = pangolin::FindPath(argv[0], "/src/shaders");
    if(shader_dir.empty()) throw std::runtime_error("Couldn't find runtime shader dir.");

    const std::string data_dir = pangolin::FindPath(argv[0], "/data");
    const std::string starmap_filename = data_dir + "/starmaps/starmap_2020_8k_combined.exr";
    const std::string image_glob = data_dir + "/image_sets/set1/DSC*.ARW";

    // Find images
    std::vector<std::string> image_filenames;
    {
        pangolin::FilesMatchingWildcard(image_glob, image_filenames);
        if(image_filenames.size() == 0) {
            std::cerr << "No images to load. Exiting." << std::endl;
            return -1;
        }
    }

    std::queue<ChannelAndInfo> to_upload;

    // load first serially to ensure order
    {
        auto channels = LoadImageAndInfo(image_filenames[0]);
        for(auto& c : channels)  to_upload.push(std::move(c));

//        // shuffle the rest for less bias incremental sampling
//        std::random_device rd;
//        std::mt19937 g(rd());
//        std::shuffle(image_filenames.begin()+1, image_filenames.end(), g);
    }

    const size_t N = image_filenames.size();
    const size_t width = to_upload.front().image.w;
    const size_t height = to_upload.front().image.h;
    const Eigen::Vector2d dims((double)width, (double)height);
    const unsigned long start_time = to_upload.front().info.timestamp;
    const float focal_mm = to_upload.front().info.focal_mm;
    const auto pix_fmt = PixelFormatFromString("GRAY16LE");

    // load rest in parallel.
    async::cancellation_token cancel_point;
    std::deque<async::task<void>> loading_tasks;
    const size_t hist_bins = 1000;

    PolarHistogram histogram[4] = { {hist_bins}, {hist_bins}, {hist_bins}, {hist_bins} };

    std::vector<std::vector<Peak>> frame_features(N);

    const double min_confidence = 2.0; //("ui.min_conf", 1.0, 0.0, 10.0);

    for(size_t i=1; i < image_filenames.size(); ++i) {
        loading_tasks.push_back(async::spawn([&,i](){
            async::interruption_point(cancel_point);
            auto channels = LoadImageAndInfo(image_filenames[i]);
            for(auto& c : channels)  to_upload.push(c);

            // Create new tasks for polar histograms (channels by value for shared_ptrs)
            loading_tasks.push_back(async::spawn([&,channels](){
                for(int c=0; c < 4; ++c) {
                    async::interruption_point(cancel_point);
                    histogram[c] += ComputePolarHistogram(channels[c].image, dims.cast<float>()/2.0f, hist_bins, 100000);
                }
            }));

            // Create new tasks for detection (channels by value for shared_ptrs)
            loading_tasks.push_back(async::spawn([&,channels](){
                async::interruption_point(cancel_point);
                std::vector<Peak> peak_vec;
                peak_vec.reserve(1000);
                DetectPeaks(peak_vec, channels[0].image, min_confidence);
                frame_features[i] = peak_vec;
            }));
        }));
    }

    CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    const int UI_WIDTH = 20* default_font().MaxWidth();
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, Attach::Pix(UI_WIDTH));
    DataLog log;
//    Plotter plot(&log);

    ImageView view_composite;
    const double aspect = width / (double)height;
    view_composite.SetAspect(aspect);
    view_composite.offset_scale.second = 10.0f;

    View view_starmap;
    view_starmap.SetAspect(aspect);

    View container;
    DisplayBase().AddDisplay(container);
    container.SetBounds(0.0, 1.0, Attach::Pix(UI_WIDTH), 1.0).SetLayout(LayoutEqual);
    container.SetHandler(new Handler());
    container.AddDisplay(view_composite).AddDisplay(view_starmap); //.AddDisplay(plot);

    pangolin::GlSlProgram prog_stack;
    prog_stack.AddShaderFromFile(pangolin::GlSlAnnotatedShader, shader_dir+"/prog_stack.glsl", {{"NUM_CP","5"}}, {shader_dir} );
    prog_stack.Link();

    pangolin::GlSlProgram prog_carree;
    prog_carree.AddShaderFromFile( pangolin::GlSlAnnotatedShader, shader_dir+"/prog_carree.glsl", {}, {shader_dir});
    prog_carree.Link();

    pangolin::GlTexture tex_starmap;
    auto img = pangolin::LoadImage(starmap_filename);
    tex_starmap.Load(img);

    for(size_t i=0; i < container.NumChildren(); ++i) {
        pangolin::RegisterKeyPressCallback('1'+i, [i,&container](){ container[i].ToggleShow(); });
    }

    double focal_pix = pixel_focal_length_from_mm(focal_mm, {36.0, 24.0}, {double(width), double(height)} );
    Eigen::Vector3d axis_angle(1.022e-5, 5.354e-5,-4.814e-5); //(0.0, 0.0, 0.0);
    double angle_offset = 1.241;

    float green_fac = 1.0;
    float red_fac = 1.0;
    float blue_fac = 1.0;
    float gamma = 1.0;
    float vig_scale = 1.0;

    Var<int> frame("ui.frame", -1, -1, N-1);
    frame.Meta().gui_changed = true;
    Var<double>::Attach("ui.f", focal_pix, width / 4.0, width * 4.0 );
    Var<double>::Attach("ui.a1", axis_angle[0], -1e-4, +1e-4);
    Var<double>::Attach("ui.a2", axis_angle[1], -1e-4, +1e-4);
    Var<double>::Attach("ui.a3", axis_angle[2], -1e-5, +1e-5);
    Var<float>::Attach("ui.green_fac", green_fac, 0.5, 1.0);
    Var<float>::Attach("ui.red_fac", red_fac, 0.5, 1.0);
    Var<float>::Attach("ui.blue_fac", blue_fac, 0.5, 1.0);
    Var<float>::Attach("ui.gamma", gamma, 0.1, 1.0);
    Var<double>::Attach("ui.angle_offset", angle_offset, -M_PI, M_PI);

    constexpr size_t spline_K = 3;
    const double control_point_interval = 400.0;
    const size_t num_control_points = int(hist_bins/control_point_interval) + spline_K;
    Eigen::MatrixXd control_points(4,num_control_points);
    control_points.setConstant(1.0);

    auto render_warped = [&](const Sophus::SO3d& R_ba, const Eigen::Matrix3d& K, const Eigen::Matrix3d& Kinv, GlTexture& tex, uint8_t bayer_channel) {
        // BGGR
        Eigen::Vector3f colors[] = {
            {red_fac, 0.0f, 0.0f},
            {0.0f, green_fac/2.0f, 0.0f},
            {0.0f, green_fac/2.0f, 0.0f},
            {0.0f, 0.0f, blue_fac},
        };

        const Eigen::Matrix3d H = K * R_ba.matrix() * Kinv;

        prog_stack.Bind();
        prog_stack.SetUniform("u_KRbaKinv", H.cast<float>().eval() );
        prog_stack.SetUniform("u_dim", dims.cast<float>().eval() );
        prog_stack.SetUniform("u_color", colors[bayer_channel] );
        prog_stack.SetUniform("u_gamma", gamma );
        prog_stack.SetUniform("u_spline_interval", (float)control_point_interval );
        glUniform1fv( prog_stack.GetUniformHandle("u_spline_control_points"), num_control_points, control_points.row(bayer_channel).cast<float>().eval().data() );
        prog_stack.SetUniform("u_spline_matrix", SplineConstants<double,spline_K>::cardinal_matrix().cast<float>().eval() );

        tex.Bind();
        glEnable(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glDrawRect(-1,-1,1,1);
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        prog_stack.Unbind();
    };

    std::vector<std::shared_ptr<TextureAndInfo>> textures;
    std::queue<std::shared_ptr<TextureAndInfo>> to_fuse;

    auto upload_next_texture = [&](){
        if(to_upload.empty()) return;

        auto channel = std::move(to_upload.front());
        to_upload.pop();
        auto fmt = GlPixFormat::FromType<uint16_t>();
        std::shared_ptr<TextureAndInfo> t( new TextureAndInfo{
            channel.info,
            GlTexture((GLint)channel.image.w, (GLint)channel.image.h, GL_LUMINANCE16, true, 0, fmt.glformat, fmt.gltype, channel.image.ptr )
        } );

        textures.push_back(t);
        if(frame == -1) {
            to_fuse.push(t);
        }
    };

    auto render_next_tex = [&]()
    {
        if(to_fuse.empty()) return;

        auto t = std::move(to_fuse.front());
        to_fuse.pop();

        const double time_s = (t->info.timestamp - start_time) / 1.0;


        const Sophus::SO3d R_ba = Sophus::SO3d::exp(time_s * axis_angle);

        Eigen::Matrix3d K_image;
        K_image << focal_pix, 0.0, width / 2.0,
             0.0, focal_pix, height / 2.0,
             0.0, 0.0, 1.0;

        Eigen::Matrix3d K_channel;
        const double dx = 0.5*(t->info.bayer_channel % 2);
        const double dy = 0.5*(t->info.bayer_channel / 2);

        // TODO: why does this only work with negatives here? Should be +ive...
        K_channel << focal_pix, 0.0, width / 2.0 - dx,
             0.0, focal_pix, height / 2.0 - dy,
             0.0, 0.0, 1.0;

        Eigen::Matrix3d K_image_inv = K_image.inverse();

        render_warped(R_ba, K_channel, K_image_inv, t->tex, t->info.bayer_channel);
    };

    view_starmap.extern_draw_function = [&](View& v){
        view_starmap.Activate();
        // Use same view as composite
        Eigen::Matrix3d K_image;
        K_image << focal_pix, 0.0, width / 2.0,
             0.0, focal_pix, height / 2.0,
             0.0, 0.0, 1.0;

        // we want a rotation which takes the polar axis in camera frame to the z-axis for J2000
        const Eigen::Vector3d polaraxis_cam = axis_angle.normalized();
        const Sophus::SO3d R_J2000_Camera = Sophus::SO3d::rotZ(angle_offset) * Sophus::SO3FromNormal(polaraxis_cam).inverse();

        const Eigen::Matrix3d RKinv = R_J2000_Camera.matrix() * K_image.inverse();

        prog_carree.Bind();
        prog_carree.SetUniform("u_RbaKinv", RKinv.cast<float>().eval() );
        prog_carree.SetUniform("u_dim", dims.cast<float>().eval() );
        prog_carree.SetUniform("u_gamma", gamma );
        prog_carree.SetUniform("u_color", red_fac, green_fac, blue_fac );


        tex_starmap.Bind();
        glEnable(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glDrawRect(-1,-1,1,1);
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        prog_carree.Unbind();
    };

    auto adjust_vignette = [&](){
        Eigen::Matrix<float,8,Eigen::Dynamic> data_to_plot(8,hist_bins);

        for(size_t c=0; c < 4; ++c) {
            if( !(histogram[c].num.array() >  0).all() ) {
                std::cerr << "histogram not adequately populated yet." << std::endl;
                return;
            }
            data_to_plot.row(c) = histogram[c].sum / histogram[c].num.cast<float>();
        }

        // fit spline
        const Eigen::Matrix<double,1,Eigen::Dynamic> rad = Eigen::VectorXd::LinSpaced(hist_bins, 0.0, double(hist_bins));

        control_points = fit_cardinal_basis_spline<double,spline_K,double>(
            num_control_points, control_point_interval, rad, data_to_plot.topRows<4>().cast<double>()
        );
        const Eigen::MatrixXd samples = eval_cardinal_basis_spline<double,spline_K,double>(
            control_point_interval, control_points, rad
        );
        data_to_plot.bottomRows<4>() = samples.cast<float>();

        log.Clear();
        log.Log(data_to_plot.rows(), data_to_plot.data(), data_to_plot.cols());
    };

    view_composite.extern_draw_function = [&](pangolin::View&){
        if( frame >= 0 && frame_features[frame].size() > 0) {
            const auto& fs = frame_features[frame];
            glPointSize(2*view_composite.GetViewScale() );
            pangolin::glDrawVertices<float>(fs.size(), &fs[0].pos[0], GL_POINTS, 2, sizeof(Peak) );
        }
    };

    const size_t view_scale = 4;
    view_composite.tex = GlTexture(view_scale*width,view_scale*height,GL_RGB32F);
    GlFramebuffer buffer(view_composite.tex);

    view_composite.SetDimensions(view_scale*width,view_scale*height);

    size_t num_fused = 0;

    pangolin::Var<std::function<void()>> do_vignette("ui.do_vignette", adjust_vignette);

    while( !pangolin::ShouldQuit() )
    {
        axis_angle = EARTH_ROTATION_RAD_PER_SECOND * axis_angle.normalized();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for(int i=0; i < 4; ++i) upload_next_texture();

        if(GuiVarHasChanged()) {
            // Restart
            view_composite.offset_scale.first = view_composite.offset_scale.first / num_fused;
            view_composite.offset_scale.second = view_composite.offset_scale.second * num_fused;

            buffer.Bind();
            glViewport(0,0,view_scale*width,view_scale*height);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            buffer.Unbind();

            while(!to_fuse.empty()) to_fuse.pop();

            if(frame == -1) {
                for(auto& t : textures) to_fuse.push(t);
            }else if( 4*frame+3 < textures.size() ) {
                for(int c=0; c < 4; ++c) {
                    to_fuse.push(textures[4*frame+c]);
                }
            }

            num_fused = 0;
        }

        const size_t to_fuse_this_it = std::min( 4ul, to_fuse.size());

        if(/*num_fused < 4 && */to_fuse_this_it) {
            buffer.Bind();
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE,GL_ONE);
            glViewport(0,0,view_scale*width,view_scale*height);

            for(size_t i=0; i < to_fuse_this_it; ++i) {
                render_next_tex();

                if(num_fused) {
                    view_composite.offset_scale.first = view_composite.offset_scale.first / num_fused * (num_fused+1);
                    view_composite.offset_scale.second = view_composite.offset_scale.second * num_fused / (num_fused+1);
                }
                ++num_fused;
            }

            glDisable(GL_BLEND);
            buffer.Unbind();

        }

        pangolin::FinishFrame();
    }

    cancel_point.cancel();
    async::when_all(loading_tasks).wait();

    return 0;
}
