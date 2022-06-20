#include "realcugan.h"
#include <cfloat>
#include <cpu.h>
#include <iostream>
#include <thread>
#include <string>
#include "fmt/core.h"

RealCUGAN::RealCUGAN() {
    std::cout << "cpu count: " << ncnn::get_big_cpu_count() << std::endl;
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    ncnnNet.opt = ncnn::Option();
    ncnnNet.opt.num_threads = ncnn::get_big_cpu_count();
}

void progress_callback(long total_cost, long tile_cost, float progress_rate)
{
    // callback by stdout =_=
    long remaining_time = 0;
    if (progress_rate != 0) {
        remaining_time = float(total_cost) / progress_rate - (float)total_cost;
    }
    std::string script = fmt::format(R"($CALLBACK$ {{"eventType":"PROC_PROGRESS","total_cost":{},"tile_cost":{},"progress_rate":{},"remaining_time":{}}})",
                                     total_cost, tile_cost, progress_rate, remaining_time);
    std::cout << script << std::endl;
}

int RealCUGAN::load(int scaleOption, int noiseOption) {
    ncnnNet.clear();
    this->scale = scaleOption;
    this->noise = noiseOption;

    std::string paramFilePath;
    std::string binFilePath;
    if (this->noise == 0) {
        paramFilePath = fmt::format("up{}x-no-denoise.param", this->scale);
        binFilePath = fmt::format("up{}x-no-denoise.bin", this->scale);
    } else if (this->noise == -1) {
        paramFilePath = fmt::format("up{}x-conservative.param", this->scale);
        binFilePath = fmt::format("up{}x-conservative.bin", this->scale);
    } else {
        paramFilePath = fmt::format("up{}x-denoise{}x.param", this->scale, this->noise);
        binFilePath = fmt::format("up{}x-denoise{}x.bin", this->scale, this->noise);
    }
    ncnnNet.load_param(paramFilePath.c_str());
    ncnnNet.load_model(binFilePath.c_str());

    if (scale == 2)
    {
        this->prepadding = 18;
    } else if (scale == 3)
    {
        this->prepadding = 14;
    } else if (scale == 4)
    {
        this->prepadding = 19;
    }
    return 0;
}

// CPU only
int RealCUGAN::process(const ncnn::Mat &inimage, ncnn::Mat &outimage) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const unsigned char *pixeldata = (const unsigned char *) inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    const int channels = inimage.elempack;

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::Option opt = ncnnNet.opt;

    // each tile 200x200
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    for (int yi = 0; yi < ytiles; yi++) {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 1 || scale == 3) {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4) {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);
        for (int xi = 0; xi < xtiles; xi++) {
            std::chrono::steady_clock::time_point tileBegin = std::chrono::steady_clock::now();

            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1 || scale == 3) {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4) {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            int in_tile_x0 = std::max(xi * TILE_SIZE_X - prepadding, 0);
            int in_tile_x1 = std::min((xi + 1) * TILE_SIZE_X + prepadding_right, w);

            // crop tile
            ncnn::Mat in;
            {
                if (channels == 3) {
                    in = ncnn::Mat::from_pixels_roi(pixeldata, ncnn::Mat::PIXEL_RGB, w, h, in_tile_x0, in_tile_y0,
                                                    in_tile_x1 - in_tile_x0, in_tile_y1 - in_tile_y0);
                }
            }
            ncnn::Mat out;

            {
                // split alpha and preproc
                ncnn::Mat in_tile;
                {
                    in_tile.create(in.w, in.h, 3);
                    for (int q = 0; q < 3; q++) {
                        const float *ptr = in.channel(q);
                        float *outptr = in_tile.channel(q);

                        for (int i = 0; i < in.w * in.h; i++) {
                            *outptr++ = *ptr++ * (1 / 255.f);
                        }
                    }
                }

                // border padding
                {
                    int pad_top = std::max(prepadding - yi * TILE_SIZE_Y, 0);
                    int pad_bottom = std::max(
                            std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom - h, prepadding_bottom), 0);
                    int pad_left = std::max(prepadding - xi * TILE_SIZE_X, 0);
                    int pad_right = std::max(std::min((xi + 1) * TILE_SIZE_X + prepadding_right - w, prepadding_right),
                                             0);

                    ncnn::Mat in_tile_padded;
                    ncnn::copy_make_border(in_tile, in_tile_padded, pad_top, pad_bottom, pad_left, pad_right, 2, 0.f,
                                           ncnnNet.opt);
                    in_tile = in_tile_padded;
                }
                // realcugan
                ncnn::Mat out_tile;
                {
                    ncnn::Extractor ex = ncnnNet.create_extractor();

                    ex.input("in0", in_tile);

                    ex.extract("out0", out_tile);
                }

                // postproc and merge alpha
                {
                    out.create(tile_w_nopad * scale, tile_h_nopad * scale, channels);
                    if (scale == 4) {
                        for (int q = 0; q < 3; q++) {
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *inptr = in_tile.channel(q).row(prepadding + i / 4) + prepadding;
                                const float *ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++) {
                                    if (outptr) {
                                        *outptr++ = *ptr++ * 255.f + 0.5f + inptr[j / 4] * 255.f;
                                    } else {
                                    }
                                }
                            }
                        }
                    } else {
                        for (int q = 0; q < 3; q++) {
                            float *outptr = out.channel(q);

                            for (int i = 0; i < out.h; i++) {
                                const float *ptr = out_tile.channel(q).row(i);

                                for (int j = 0; j < out.w; j++) {
                                    *outptr++ = *ptr++ * 255.f + 0.5f;
                                }
                            }
                        }
                    }
                }
            }
            {
                if (channels == 3) {
                    out.to_pixels((unsigned char *) outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels +
                                  xi * scale * TILE_SIZE_X * channels, ncnn::Mat::PIXEL_RGB, w * scale * channels);
                }
            }

            auto end = std::chrono::steady_clock::now();
            auto tile_cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - tileBegin).count();
            auto total_cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            float progress_rate = (float) (xtiles * yi + xi + 1) / (float) (xtiles * ytiles);
            progress_callback(total_cost, tile_cost, progress_rate);
        }
    }

    return 0;
}
