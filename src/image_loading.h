#pragma once

#include <libraw/libraw.h>
#include <pangolin/image/managed_image.h>
#include <pangolin/gl/gl.h>


struct ImageInfo
{
    long timestamp;
    float focal_mm;
    uint8_t bayer_channel;
};

struct ChannelAndInfo
{
    ImageInfo info;
    pangolin::ManagedImage<uint16_t> image;
};

struct TextureAndInfo
{
    ImageInfo info;
    pangolin::GlTexture tex;
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
