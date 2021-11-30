#include <pangolin/gl/gldraw.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/display/widgets.h>
#include <pangolin/display/default_font.h>
#include <pangolin/display/image_view.h>
#include <pangolin/var/var.h>
#include <pangolin/video/video.h>
#include <pangolin/utils/file_utils.h>

#include <sophus/so3.hpp>

#include <libraw/libraw.h>
#include <async++.h>

#include <unordered_map>
#include <deque>
#include <thread>
#include <future>
#include <algorithm>
#include <queue>
#include <cmath>

using namespace pangolin;

const std::string star_shader = R"Shader(
@start vertex
#version 120
attribute vec3 a_position;
varying vec2 v_pos;

void main() {
    gl_Position = vec4(a_position, 1.0);
    v_pos = a_position.xy * vec2(0.5,-0.5) + vec2(0.5,0.5);
}

@start fragment
#version 120
varying vec2 v_pos;
uniform vec2 u_dim;
uniform vec3 u_color;
uniform mat3 u_KRbaKinv;
uniform sampler2D tex;

vec3 Unproject(vec2 p)
{
    return vec3(p, 1.0);
}

vec2 Project(vec3 p)
{
    return vec2(p.x/p.z, p.y/p.z);
}

vec2 Pix2Tex(vec2 p, vec2 dim)
{
    return (p + vec2(0.5)) / dim;
}

void main() {
    vec2 Pa = v_pos * u_dim - vec2(0.5);
    vec2 Pb = Project(u_KRbaKinv * Unproject(Pa));
    float I = texture2D(tex,Pix2Tex(Pb,u_dim)).x;
//    float I = texture2D(tex,v_pos).x;
    gl_FragColor = vec4(I*u_color, 1.0);
}
)Shader";

double pixel_focal_length_from_mm(double focal_length_mm, const Eigen::Vector2d& sensor_dim_mm, const Eigen::Vector2d& image_dim_pix)
{
    const Eigen::Vector2d pix_per_mm = image_dim_pix.array() / sensor_dim_mm.array();
    std::cout << pix_per_mm.transpose() << std::endl;
    return focal_length_mm * pix_per_mm[0];
}

struct ImageInfo
{
    long timestamp;
    float focal_mm;
    uint8_t bayer_channel;
};

struct ChannelAndInfo
{
    ImageInfo info;
    ManagedImage<uint16_t> image;
};

struct TextureAndInfo
{
    ImageInfo info;
    GlTexture tex;
};

std::array<ChannelAndInfo,4> LoadImageAndInfo(const std::string& filename)
{
    auto raw = std::make_unique<LibRaw>();

    int result;

    if ((result = raw->open_file(filename.c_str())) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(libraw_strerror(result));
    }

    if ((result = raw->unpack()) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(libraw_strerror(result));
    }

    const auto& S = raw->imgdata.sizes;

    std::array<ChannelAndInfo,4> ret;
    for(size_t c=0; c < 4; ++c) {
        ret[c].info.timestamp = raw->imgdata.other.timestamp;
        ret[c].info.focal_mm = raw->imgdata.other.focal_len;
        ret[c].info.bayer_channel = c;
        auto& img_out = ret[c].image;
        img_out.Reinitialise(S.width / 2, S.height / 2);
        for(unsigned y=0; y < img_out.h; ++y) {
            uint16_t* outp = img_out.RowPtr(y);
            uint16_t* endp = img_out.RowPtr(y) + img_out.w;
            uint16_t* inp = raw->imgdata.rawdata.raw_image + (c%2) + (2*y+c/2)*S.raw_width;
            while(outp != endp) {
                *outp = *inp;
                inp+=2;
                outp++;
            }
        }
    }
    return ret;
}

int main( int /*argc*/, char** /*argv*/ )
{
    const std::string path = "/Users/stevenlovegrove/code/telescope/data/DSC*.ARW";
    std::vector<std::string> image_filenames;
    {
        pangolin::FilesMatchingWildcard(path, image_filenames);
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

        // shuffle the rest for less bias incremental sampling
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(image_filenames.begin()+1, image_filenames.end(), g);
    }

    // load rest in parallel.
    async::cancellation_token cancel_point;
    std::vector<async::task<void>> loading_tasks;
    for(size_t i=1; i < image_filenames.size(); ++i) {
        loading_tasks.push_back(async::spawn([&,i](){
            async::interruption_point(cancel_point);
            auto channels = LoadImageAndInfo(image_filenames[i]);
            for(auto& c : channels)  to_upload.push(std::move(c));
        }));
    }

    const size_t N = image_filenames.size();
    const size_t width = to_upload.front().image.w;
    const size_t height = to_upload.front().image.h;
    const unsigned long start_time = to_upload.front().info.timestamp;
    const float focal_mm = to_upload.front().info.focal_mm;
    const auto pix_fmt = PixelFormatFromString("GRAY16LE");

    CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    const int UI_WIDTH = 20* default_font().MaxWidth();
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, Attach::Pix(UI_WIDTH));

    ImageView view;
    const double aspect = width / (double)height;
    view.SetAspect(aspect);
    view.offset_scale.second = 10.0f;

    View container;
    DisplayBase().AddDisplay(container);
    container.SetBounds(0.0, 1.0, Attach::Pix(UI_WIDTH), 1.0).SetLayout(LayoutEqual);
    container.AddDisplay(view)/*.AddDisplay(view2)*/.SetHandler(new Handler());

    pangolin::GlSlProgram prog;
    prog.AddShader( pangolin::GlSlAnnotatedShader, star_shader );
    prog.Link();

    for(size_t i=0; i < container.NumChildren(); ++i) {
        pangolin::RegisterKeyPressCallback('1'+i, [i,&container](){ container[i].ToggleShow(); });
    }

    Var<int> frame("ui.frame", 0, 0, N-1);
    frame.Meta().gui_changed = true;

    std::cout << focal_mm << std::endl;
    std::cout << width << " x " << height << std::endl;
    double focal_pix = pixel_focal_length_from_mm(focal_mm, {36.0, 24.0}, {double(width), double(height)} );
    Eigen::Vector3d axis_angle(1.064e-5, 5.311e-5,-6.526e-5); //(0.0, 0.0, 0.0);

    Var<double>::Attach("ui.f", focal_pix, width / 4.0, width * 4.0 );
    Var<double>::Attach("ui.a1", axis_angle[0], -1e-4, +1e-4);
    Var<double>::Attach("ui.a2", axis_angle[1], -1e-4, +1e-4);
    Var<double>::Attach("ui.a3", axis_angle[2], -1e-5, +1e-5);

    Eigen::Vector2f offset_scale(0.0f, 1.0f);

    auto render_warped = [&](const Sophus::SO3d& R_ba, const Eigen::Matrix3d& K, const Eigen::Matrix3d& Kinv, GlTexture& tex, uint8_t bayer_channel) {
        const Eigen::Vector2d dims((double)width, (double)height);

        // BGGR
        Eigen::Vector3f colors[] = {
            {0.0f, 0.0f, 1.0f},
            {0.0f, 0.5f, 0.0f},
            {0.0f, 0.5f, 0.0f},
            {1.0f, 0.0f, 0.0f},
        };

        const Eigen::Matrix3d H = K * R_ba.matrix() * Kinv;

        prog.Bind();
        prog.SetUniform("u_KRbaKinv", H.cast<float>().eval() );
        prog.SetUniform("u_dim", dims.cast<float>().eval() );
        prog.SetUniform("u_color", colors[bayer_channel] );

        tex.Bind();
        glEnable(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glDrawRect(-1,-1,1,1);
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        prog.Unbind();
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
        to_fuse.push(t);
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
        K_channel << focal_pix, 0.0, width / 2.0 + dx,
             0.0, focal_pix, height / 2.0 + dy,
             0.0, 0.0, 1.0;

        Eigen::Matrix3d K_channel_inv = K_channel.inverse();

        render_warped(R_ba, K_image, K_channel_inv, t->tex, t->info.bayer_channel);
    };

    const size_t view_scale = 4;
    view.tex = GlTexture(view_scale*width,view_scale*height,GL_RGB32F);
    GlFramebuffer buffer(view.tex);

    view.SetDimensions(view_scale*width,view_scale*height);

    size_t num_fused = 0;

    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for(int i=0; i < 4; ++i) upload_next_texture();

        if(GuiVarHasChanged()) {
            // Restart
            view.offset_scale.first = view.offset_scale.first / num_fused;
            view.offset_scale.second = view.offset_scale.second * num_fused;

            buffer.Bind();
            glViewport(0,0,view_scale*width,view_scale*height);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            buffer.Unbind();

            while(!to_fuse.empty()) to_fuse.pop();
            for(auto& t : textures) to_fuse.push(t);
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
                    view.offset_scale.first = view.offset_scale.first / num_fused * (num_fused+1);
                    view.offset_scale.second = view.offset_scale.second * num_fused / (num_fused+1);
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
