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
#include "memo.hpp"

#include <libraw/libraw.h>

#include <unordered_map>
#include <deque>
#include <thread>
#include <future>

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
uniform vec2 u_offset_scale;
uniform vec2 u_dim;
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

void main() {
    vec2 Pa = v_pos * u_dim;
    vec2 Pb = Project(u_KRbaKinv * Unproject(Pa));

    vec3 color = texture2D(tex,Pb/u_dim).xyz;
    color += vec3(u_offset_scale.x, u_offset_scale.x, u_offset_scale.x);
    color *= u_offset_scale.y;
    gl_FragColor = vec4(color, 1.0);
}
)Shader";

double pixel_focal_length_from_mm(double focal_length_mm, const Eigen::Vector2d& sensor_dim_mm, const Eigen::Vector2d& image_dim_pix)
{
    const Eigen::Vector2d pix_per_mm = image_dim_pix.array() / sensor_dim_mm.array();
    std::cout << pix_per_mm.transpose() << std::endl;
    return focal_length_mm * pix_per_mm[0];
}

struct ImageAndInfo
{
    pangolin::TypedImage image;
    long timestamp;
};

ImageAndInfo LoadImageAndInfo(const std::string& filename)
{
    using namespace pangolin;

    LibRaw RawProcessor;

    int result;

    if ((result = RawProcessor.open_file(filename.c_str())) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(libraw_strerror(result));
    }

    if ((result = RawProcessor.unpack()) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(libraw_strerror(result));
    }

    const auto& S = RawProcessor.imgdata.sizes;

    ImageAndInfo ret;
    ret.timestamp = RawProcessor.imgdata.other.timestamp;
    ret.image.Reinitialise(S.width, S.height, PixelFormatFromString("GRAY16LE"), sizeof(uint16_t) * S.raw_width);
    PitchedCopy((char*)ret.image.ptr, ret.image.pitch, (char*)RawProcessor.imgdata.rawdata.raw_image, sizeof(uint16_t) * S.raw_width, sizeof(uint16_t) * ret.image.w, ret.image.h);
    return ret;
}

ImageAndInfo LoadImageAndInfo(char* buffer, size_t size_bytes)
{
    using namespace pangolin;

    LibRaw RawProcessor;

    int result;

    if ((result = RawProcessor.open_buffer(buffer, size_bytes)) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(libraw_strerror(result));
    }

    if ((result = RawProcessor.unpack()) != LIBRAW_SUCCESS)
    {
        throw std::runtime_error(libraw_strerror(result));
    }

    const auto& S = RawProcessor.imgdata.sizes;

    ImageAndInfo ret;
    ret.timestamp = RawProcessor.imgdata.other.timestamp;
    ret.image.Reinitialise(S.width, S.height, PixelFormatFromString("GRAY16LE"), sizeof(uint16_t) * S.raw_width);
    PitchedCopy((char*)ret.image.ptr, ret.image.pitch, (char*)RawProcessor.imgdata.rawdata.raw_image, sizeof(uint16_t) * S.raw_width, sizeof(uint16_t) * ret.image.w, ret.image.h);
    return ret;
}

template<typename R>
  bool is_ready(std::future<R> const& f)
  { return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }


int main( int /*argc*/, char** /*argv*/ )
{
    LibRaw raw[11];
    return 0;

    using namespace pangolin;

    const std::string path = "/Users/stevenlovegrove/code/telescope/data/DSC*.ARW";
    std::vector<ImageAndInfo> loaded_images;
    loaded_images.reserve(1000);

    {
        std::vector<std::string> image_filenames;
        FilesMatchingWildcard(path, image_filenames);

        for(const auto& filename : image_filenames)
        {
            std::cout << filename << std::endl;
            std::string s = GetFileContents(filename);
//            loaded_images.push_back(LoadImageAndInfo(s.data(), s.size()));
//            loaded_images.push_back(LoadImageAndInfo(filename));
        }
    }

    return 0;

    auto video = OpenVideo("debayer:[tile=rggb,method=mono]///Users/stevenlovegrove/code/telescope/data/DSC*.ARW");
    const size_t width = video->Streams()[0].Width();
    const size_t height = video->Streams()[0].Height();
    const size_t pitch = video->Streams()[0].Pitch();
    const auto pix_fmt = video->Streams()[0].PixFormat();

    auto* playback = FindFirstMatchingVideoInterface<VideoPlaybackInterface>(*video.get());
    if(!playback) {
        std::cerr << "Only supports playback video" << std::endl;
        return -1;
    }

    CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    const int UI_WIDTH = 20* default_font().MaxWidth();
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, Attach::Pix(UI_WIDTH));

    ImageView view;
//    View view2;
    const double aspect = width / (double)height;
    view.SetAspect(aspect);
//    view2.SetAspect(aspect);
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

    Var<int> frame("ui.frame", 0, 0, playback->GetTotalFrames()-1);
    frame.Meta().gui_changed = true;

    double f = 2963.0; //video->Streams()[0].Width() / 2.0;
    Eigen::Vector3d axis_angle(2.234e-5, 0.0001174,-9.798e-5); //(0.0, 0.0, 0.0);

    Var<double>::Attach("ui.f", f, width / 4.0, width * 4.0 );
    Var<double>::Attach("ui.a1", axis_angle[0], -1e-4, +1e-4);
    Var<double>::Attach("ui.a2", axis_angle[1], -1e-4, +1e-4);
    Var<double>::Attach("ui.a3", axis_angle[2], -1e-5, +1e-5);

    auto load_image = [&](size_t frame) {
        playback->Seek(frame);
        TypedImage image(width, height, pix_fmt, pitch);
        video->GrabNext(image.ptr);
        return image;
    };

    auto get_image = memo::memoize(load_image);

    Eigen::Vector2f offset_scale(0.0f, 1.0f);

    auto render_warped = [&](const Sophus::SO3d& R_ba, Eigen::Matrix3d& K, GlTexture& tex) {
        const Eigen::Vector2d dims((double)width, (double)height);

        prog.Bind();
        prog.SetUniform("u_offset_scale", offset_scale);
        prog.SetUniform("u_KRbaKinv", (K * R_ba.matrix() * K.inverse()).cast<float>().eval() );
        prog.SetUniform("u_dim", dims.cast<float>().eval() );

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        tex.Bind();
        glEnable(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glDrawRect(-1,-1,1,1);
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        prog.Unbind();
    };

    std::unordered_map<size_t, GlTexture> textures;

    auto get_tex = [&](size_t frame) -> GlTexture& {
        auto it = textures.find(frame);
        if(it != textures.end()) {
            return it->second;
        }else{
            auto it_res = textures.emplace(std::make_pair(frame, GlTexture(get_image(frame))));
            return it_res.first->second;
        }
    };

    auto render_frame = [&](size_t frame)
    {
        const Sophus::SO3d R_ba = Sophus::SO3d::exp(frame * axis_angle);
        Eigen::Matrix3d K;
        K << f, 0.0, width / 2.0,
             0.0, f, height / 2.0,
             0.0, 0.0, 1.0;

        render_warped(R_ba, K, get_tex(frame));
    };

    view.tex = GlTexture(width,height,GL_RGB32F);
    GlFramebuffer buffer(view.tex);

//    const float h_fov_rad = 2.0*atan2(width/2.0, f);
//    std::cout <<  "Horizontal field of view: " << 180.0*h_fov_rad/M_PI << std::endl;
//    std::cout << pixel_focal_length_from_mm(55.0, {36,24}, {(double)width,(double)height}) << std::endl;

    size_t to_fuse = 0;

    view.SetDimensions(width,height);

    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(GuiVarHasChanged()) {
            view.offset_scale.first = view.offset_scale.first / to_fuse;
            view.offset_scale.second = view.offset_scale.second * to_fuse;

            to_fuse = 0;
            buffer.Bind();
            glViewport(0,0,width,height);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            buffer.Unbind();
        }

        if(to_fuse < playback->GetTotalFrames()) {
            buffer.Bind();
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE,GL_ONE);
            glViewport(0,0,width,height);
            render_frame(to_fuse);
            glDisable(GL_BLEND);
            buffer.Unbind();

            if(to_fuse) {
                view.offset_scale.first = view.offset_scale.first / to_fuse * (to_fuse+1);
                view.offset_scale.second = view.offset_scale.second * to_fuse / (to_fuse+1);
            }
            ++to_fuse;
        }

        pangolin::FinishFrame();
    }

    return 0;
}
