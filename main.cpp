#include <stdio.h>
#include <iostream>
#include "realcugan.h"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include <chrono>
#include <thread>
#include "realcugan.h"
#include <cpu.h>
#include "fmt/core.h"

//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "stb_image_write.h"

class Task {
public:
    int image_id;
    unsigned char *input_image_data;
    unsigned char *output_image_data;
    int input_w;
    int input_h;
    int input_channel;
    int scale;
    int noise;
};

class ImageInfo
{
public:
    unsigned char* image_data;
    int channel;
    int w;
    int h;
};

static ncnn::Mutex lock;
static ncnn::ConditionVariable condition;
static RealCUGAN *realcugan;
static Task *proc_img_task;

static ncnn::Mutex finish_lock;
static ncnn::ConditionVariable finish_condition;

void remove_alpha_channel(unsigned char *image_data, int w, int h) {
    // remove the alpha channel to avoid exceeding memory limits
    for (int i = 0; i < w * h; i++) {
        image_data[i * 3] = image_data[i * 4];
        image_data[i * 3 + 1] = image_data[i * 4 + 1];
        image_data[i * 3 + 2] = image_data[i * 4 + 2];
    }
}

void copy_with_alpha_channel(unsigned char *dst, const unsigned char *src, int w, int h) {
    for (int i = 0; i < w * h; i++) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 255;
        dst += 4;
        src += 3;
    }
}

void process_image_success_callback(int image_id, long cost)
{
    // callback by stdout =_=
    std::string script = fmt::format(R"($CALLBACK$ {{"eventType": "PROC_END", "image_id": {}, "cost": {}}})", image_id, cost);
    std::cout << script << std::endl;
}

static void worker() {
    while (1) {
        lock.lock();
        while (proc_img_task == nullptr) {
            condition.wait(lock);
        }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::cout << "thread process start" << std::endl;

        if (!realcugan) {
            realcugan = new RealCUGAN();
            realcugan->load(proc_img_task->scale, proc_img_task->noise);
        }
        if (realcugan->scale != proc_img_task->scale || realcugan->noise != proc_img_task->noise)
        {
            realcugan->load(proc_img_task->scale, proc_img_task->noise);
        }

        ncnn::Mat inImage = ncnn::Mat(proc_img_task->input_w, proc_img_task->input_h,
                                      (void *) proc_img_task->input_image_data, (size_t) proc_img_task->input_channel,
                                      proc_img_task->input_channel);
        ncnn::Mat outImage = ncnn::Mat(inImage.w * realcugan->scale, inImage.h * realcugan->scale,
                                       (size_t) inImage.elemsize, (int) inImage.elemsize);
        realcugan->process(inImage, outImage);
        copy_with_alpha_channel(proc_img_task->output_image_data, (const unsigned char *) outImage.data, outImage.w,
                                outImage.h);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "thread process done, cost: " << cost / 1000 << "secs" << std::endl;
        process_image_success_callback(proc_img_task->image_id, cost);

        delete proc_img_task;
        proc_img_task = nullptr;
        lock.unlock();
        finish_lock.lock();
        finish_condition.signal();
        finish_lock.unlock();

    }
}

static std::thread t(worker);

extern "C"
{

int process_image(int image_id, unsigned char *input_image_data, unsigned char *output_image_data, int input_w,
                  int input_h, int scale, int noise) {
    lock.lock();

    if (proc_img_task != nullptr) {
        return -1;
    }
    remove_alpha_channel(input_image_data, input_w, input_h);

    Task *tsk = new Task();
    tsk->image_id = image_id;
    tsk->input_image_data = input_image_data;
    tsk->output_image_data = output_image_data;
    tsk->input_w = input_w;
    tsk->input_h = input_h;
    tsk->input_channel = 3;
    tsk->scale = scale;
    tsk->noise = noise;
    proc_img_task = tsk;

    lock.unlock();
    condition.signal();
    return 0;
}

}