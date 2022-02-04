#include<opencv2/opencv.hpp>
#include<nce_alg.hpp>
#include<alg_type.h>
#include<memory>

using namespace nce_alg;
using namespace cv;
using namespace std;

#if (defined WIN32)
#include <time.h>
#define OSA_DEBUG_DEFINE_TIME \
    clock_t start;            \
    clock_t end;
#else
#include <sys/time.h>
#define OSA_DEBUG_DEFINE_TIME \
    struct timespec start;    \
    struct timespec end;
#endif

#if (defined WIN32)
#define OSA_DEBUG_START_TIME start = clock();
#else
#define OSA_DEBUG_START_TIME clock_gettime(CLOCK_REALTIME, &start);
#endif

#if (defined WIN32)
#define OSA_DEBUG_END_TIME(S) \
    end = clock();            \
    printf("%s %ld ms\n", #S, end - start);
#else
#define OSA_DEBUG_END_TIME(S)            \
    clock_gettime(CLOCK_REALTIME, &end); \
    printf("%s %ld ms\n", #S, 1000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000);
#endif

typedef struct FaceResults
{
    vector<detect_result> face_location;
    vector<landmarks>     face_landmarks;
};

class FaceDetect
{
public:
    FaceDetect(const char* config_file, const char* landmark_config)
    {
        p_detect_machine = std::unique_ptr<nce_alg_machine>(new nce_alg_machine(RETINAFACE, MNNPLATFORM));
        p_detect_machine->nce_alg_init(config_file, input_infos);
        p_landmark_machine = std::unique_ptr<nce_alg_machine>(new nce_alg_machine(THREE_DDFA, MNNPLATFORM));
        p_landmark_machine->nce_alg_init(landmark_config, tddfa_infos);

        img_t tmp_img;
        img_t tmp_tddfa_img;

        tmp_img.image_attr = input_infos[0];
        tmp_tddfa_img.image_attr = tddfa_infos[0];

        tddfa_imgs.push_back(tmp_tddfa_img);
        input_imgs.push_back(tmp_img);
    }

    int get_face(Mat& frame, FaceResults & face_results)
    {
        float factor_x = (float)frame.cols / input_infos[0].u32Width;
        float factor_y = (float)frame.rows / input_infos[0].u32Height;

        resize(frame, input_frame, Size(input_infos[0].u32Width, input_infos[0].u32Height));
        cvtColor(input_frame, input_frame, COLOR_BGR2RGB);
        input_imgs[0].image = input_frame.data;

        p_detect_machine->nce_alg_inference(input_imgs);
        p_detect_machine->nce_alg_get_result(alg_results);

        for (int i = 0; i < alg_results.num; i++)
        {
            detect_result* result = (detect_result*)((alg_results.st_alg_results + i)->obj);
            result->x1 = result->x1 * factor_x;
            result->y1 = result->y1 * factor_y;
            result->x2 = result->x2 * factor_x;
            result->y2 = result->y2 * factor_y;
            face_results.face_location.push_back(*result);
        }

        get_landmarks(frame, face_results);
        return NCE_SUCCESS;
    }

    int get_landmarks(Mat frame, FaceResults & face_results)
    {
        for (auto box : face_results.face_location)
        {
            float w = box.x2 - box.x1;
            float h = box.y2 - box.y1;
            box.y2 = box.y1 + h * 1.1;
            box.x1 = max(0., box.x1 - 0.2 * w);
            box.x2 = max(0., box.x2 + 0.2 * w);

            w = box.x2 - box.x1;
            h = box.y2 - box.y1;
            w = min((float)frame.cols - box.x1, w);
            h = min((float)frame.rows - box.y1, h);
            //float ctx = (box.x1 + box.x2) / 2;
            //float cty = (box.y1 + box.y2) / 2;
            //float radius = max(w, h) / 2;
            //box.x1 = max(0.f, ctx - radius);
            //box.y1 = max(0.f, cty - radius);
            //w = min((float)frame.cols - box.x1, 2 * radius);
            //h = min((float)frame.cols - box.x1, 2 * radius);

            float factor_x = (float)w / tddfa_infos[0].u32Width;
            float factor_y = (float)h / tddfa_infos[0].u32Height;

            Mat roi_img(frame, Rect(box.x1, box.y1, w, h));
            resize(roi_img, tddfa_frame, Size(tddfa_infos[0].u32Width, tddfa_infos[0].u32Height));
            imshow("roi_img", roi_img);
            cvtColor(tddfa_frame, tddfa_frame, COLOR_BGR2RGB);
            tddfa_imgs[0].image = tddfa_frame.data;

            p_landmark_machine->nce_alg_inference(tddfa_imgs);
            p_landmark_machine->nce_alg_get_result(alg_results);


            landmarks* result = (landmarks*)(alg_results.st_alg_results->obj);
            for (int i = 0; i < result->dims; i++)
            {
                (result->points + i)->x = (result->points + i)->x * factor_x + box.x1;
                (result->points + i)->y = (result->points + i)->y * factor_y + box.y1;
            }
            face_results.face_landmarks.push_back(*result);
        }
        return NCE_SUCCESS;
    }

    int draw_bboxes(Mat & frame, FaceResults & face_results)
    {
        char str[20];
        for (int i = 0; i < face_results.face_location.size(); i++)
        {
            int x1 = face_results.face_location[i].x1;
            int x2 = face_results.face_location[i].x2;
            int y1 = face_results.face_location[i].y1;
            int y2 = face_results.face_location[i].y2;
            
            float score = face_results.face_location[i].score;

            for (int k = 0; k < 5; k++)
            {
                int x = (face_results.face_landmarks[i].points + k)->x;
                int y = (face_results.face_landmarks[i].points + k)->y;
                circle(frame, Point(x, y), 1, Scalar(0, 0, 255));
            }

            for (int j = 0; j < 68; j++)
            {
                int x = (face_results.face_landmarks[i].points + j)->x;
                int y = (face_results.face_landmarks[i].points + j)->y;
                circle(frame, Point(x, y), 1, Scalar(0, 0, 255));
            }
            sprintf(str, "score: %4f", score);
            rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 255));
            
            putText(frame, str, Point(x1, y1), 0, 1, Scalar(255, 255, 255));
        }
        return NCE_SUCCESS;
    }

    int detect_destroy()
    {
        p_detect_machine->nce_alg_destroy();
        return NCE_SUCCESS;
    }
private:
    unique_ptr<nce_alg_machine> p_detect_machine;
    unique_ptr<nce_alg_machine> p_landmark_machine;
    alg_result_info alg_results;
    vector<img_info> input_infos;
    vector<img_info> tddfa_infos;

    vector<img_t> input_imgs;
    vector<img_t> tddfa_imgs;
    Mat input_frame;
    Mat tddfa_frame;
};



int main(int argc, char* argv[])
{
    char* retinaface_path = argv[1];
    char* tddfa_path = argv[2];
    OSA_DEBUG_DEFINE_TIME;
    VideoCapture cap(0);
    Mat frame;


    try
    {
        FaceDetect face_detect(retinaface_path, tddfa_path);
        while (1)
        {
            if (!cap.read(frame))
            {
                printf("open camera failed! please check your camera");
            };
            FaceResults face_results;
            OSA_DEBUG_START_TIME
            face_detect.get_face(frame, face_results);
            OSA_DEBUG_END_TIME(infer cost:)
            face_detect.draw_bboxes(frame, face_results);
            cv::imshow("frame", frame);
            cv::waitKey(1);
        }
    }

    catch (const std::exception& error)
    {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return EXIT_FAILURE;
    }
}