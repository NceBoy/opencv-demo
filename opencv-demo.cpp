#include<opencv2/opencv.hpp>
#include<nce_alg.hpp>
#include<alg_type.h>
#include<memory>
#include<common.h>
#include<assert.h>

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
    typedef struct FaceName 
    {
        float cos_sim;
        string name;
    };
    vector<detect_result> face_location;
    vector<landmarks>     face_landmarks;
    vector<FaceName>        face_name;
};

int draw_bboxes(Mat& frame, FaceResults& face_results)
{
    char score_str[20];
    char sim_str[20];
    char name_str[20];
    for (int i = 0; i < face_results.face_location.size(); i++)
    {
        int x1 = face_results.face_location[i].x1;
        int x2 = face_results.face_location[i].x2;
        int y1 = face_results.face_location[i].y1;
        int y2 = face_results.face_location[i].y2;

        float score = face_results.face_location[i].score;

        for (int k = 0; k < 5; k++)
        {
            int x = (face_results.face_location[i].landmark[2 * k]);
            int y = (face_results.face_location[i].landmark[2 * k + 1]);
            circle(frame, Point(x, y), 1, Scalar(255, 255, 255));
        }

        //for (int j = 0; j < 68; j++)
        //{
        //    int x = (face_results.face_landmarks[i].points + j)->x;
        //    int y = (face_results.face_landmarks[i].points + j)->y;
        //    circle(frame, Point(x, y), 1, Scalar(0, 0, 255));
        //}
        auto name = face_results.face_name[i].name.c_str();
        auto cos_sim = face_results.face_name[i].cos_sim;
        sprintf(score_str, "det score: %4f", score);
        sprintf(sim_str, "cos_sim: %4f", cos_sim);
        sprintf(name_str, "name: %s", name);
        rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 255));
        putText(frame, score_str, Point(x1, y1), 0, 1, Scalar(255, 255, 255));
        putText(frame, sim_str, Point(x1, y1 + 30), 0, 1, Scalar(255, 0, 0));
        putText(frame, name_str, Point(x1, y1 + 60), 0, 1, Scalar(0, 0, 255));
    }
    return NCE_SUCCESS;
}

class FaceDetect
{
public:
    vector<img_info> input_infos;
    FaceDetect(const char* config_file, const char* landmark_config)
    {
        p_detect_machine = std::unique_ptr<nce_alg_machine>(new nce_alg_machine(RETINAFACE, MNNPLATFORM));
        p_detect_machine->nce_alg_init(config_file, input_infos);


        img_t tmp_img;


        tmp_img.image_attr = input_infos[0];
        //p_landmark_machine = std::unique_ptr<nce_alg_machine>(new nce_alg_machine(THREE_DDFA, MNNPLATFORM));
        //p_landmark_machine->nce_alg_init(landmark_config, tddfa_infos);
        //img_t tmp_tddfa_img;
        //tmp_tddfa_img.image_attr = tddfa_infos[0];
        //tddfa_imgs.push_back(tmp_tddfa_img);
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
            for (int k = 0; k < 5; k++)
            {
                result->landmark[2*k] = result->landmark[2 * k] * factor_x;
                result->landmark[2 * k + 1] = result->landmark[2 * k + 1] * factor_y;
            }
            face_results.face_location.push_back(*result);
            face_results.face_name.push_back({0.f, "Unknown" });
        }

        //get_landmarks(frame, face_results);
        return NCE_SUCCESS;
    }
#if 0 
    int get_landmarks(Mat frame, FaceResults & face_results)
    {
        for (auto box : face_results.face_location)
        {
            float w = box.x2 - box.x1;
            float h = box.y2 - box.y1;

            box.x1 = max((NCE_S32)box.x1, 0);
            box.y1 = max((NCE_S32)box.y1, 0);

            w = min(w, float(frame.cols - box.x1) - 1);
            h = min(h, float(frame.rows - box.y1) - 1);

            //box.y2 = box.y1 + h * 1.1;
            //box.x1 = max(0., box.x1 - 0.2 * w);
            //box.x2 = max(0., box.x2 + 0.2 * w);

            //w = box.x2 - box.x1;
            //h = box.y2 - box.y1;
            //w = min((float)frame.cols - box.x1, w);
            //h = min((float)frame.rows - box.y1, h);
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
#endif
    int detect_destroy()
    {
        p_detect_machine->nce_alg_destroy();
        return NCE_SUCCESS;
    }
private:
    unique_ptr<nce_alg_machine> p_detect_machine;
    //unique_ptr<nce_alg_machine> p_landmark_machine;
    alg_result_info alg_results;

    //vector<img_info> tddfa_infos;

    vector<img_t> input_imgs;
    //vector<img_t> tddfa_imgs;
    Mat input_frame;
    //Mat tddfa_frame;
};

class FaceRec
{
public:
    float sim_thresh = 0.7;
    FaceRec(const char* yaml_path, const char* image_name, FaceDetect& detect_module)
    {

        char posfix[20];
        sscanf(image_name, "%[^.]%s", name, posfix);

        p_rec_machine = std::unique_ptr<nce_alg_machine>(new nce_alg_machine(ARC_FACE, MNNPLATFORM));
        p_rec_machine->nce_alg_init(yaml_path, input_infos);

        target_points[0].x = 38.2946;
        target_points[0].y = 51.6963;
        target_points[1].x = 73.5318;
        target_points[1].y = 51.5014;
        target_points[2].x = 56.0252;
        target_points[2].y = 71.7366;
        target_points[3].x = 41.5493;
        target_points[3].y = 92.3655;
        target_points[4].x = 70.7299;
        target_points[4].y = 92.2041;

        FaceResults tmp_face_result;
        Mat ori_img = imread(image_name);
        detect_module.get_face(ori_img, tmp_face_result);
        assert(tmp_face_result.face_location.size() == 1);
        detect_result tmp_box = tmp_face_result.face_location[0];
        point cur_points[5];
        cur_points[0].x = tmp_box.landmark[0];
        cur_points[0].y = tmp_box.landmark[1];
        cur_points[1].x = tmp_box.landmark[2];
        cur_points[1].y = tmp_box.landmark[3];
        cur_points[2].x = tmp_box.landmark[4];
        cur_points[2].y = tmp_box.landmark[5];
        cur_points[3].x = tmp_box.landmark[6];
        cur_points[3].y = tmp_box.landmark[7];
        cur_points[4].x = tmp_box.landmark[8];
        cur_points[4].y = tmp_box.landmark[9];


        label_img = Mat(input_infos[0].u32Width, input_infos[0].u32Height, CV_8UC3);


        img_t tmp_img_ori;
        tmp_img_ori.image                 = ori_img.data;
        tmp_img_ori.image_attr.u32Width   = ori_img.cols;
        tmp_img_ori.image_attr.u32Height  = ori_img.rows;
        tmp_img_ori.image_attr.u32channel = 3;

        img_t tmp_img;
        tmp_img.image_attr = input_infos[0];
        tmp_img.image = label_img.data;

        get_roi(cur_points, tmp_img, tmp_img_ori, tmp_box);
        //get_inv_affine_matrix(cur_points, target_points, affine_matrix);
        //inv_warp_affine(tmp_img_ori, tmp_img, affine_matrix);

        input_imgs.push_back(tmp_img);

        p_rec_machine->nce_alg_inference(input_imgs);
        p_rec_machine->nce_alg_get_result(tmp_results);
        FaceID* tmp_label_id = (FaceID*)(tmp_results.st_alg_results->obj);
        label_id.dims = tmp_label_id->dims;
        label_id.face_id = new NCE_F32[label_id.dims];
        memcpy(label_id.face_id, tmp_label_id->face_id, sizeof(float) * label_id.dims);
    }

    ~FaceRec()
    {
        delete label_id.face_id;
    }
    int compare_face(Mat & frame, FaceResults & face_results)
    {
        memset(label_img.data, 0, label_img.cols * label_img.rows * 3);
        img_t tmp_input_img;
        tmp_input_img.image_attr.u32Width = frame.cols;
        tmp_input_img.image_attr.u32Height = frame.rows;
        tmp_input_img.image_attr.u32channel = 3;
        tmp_input_img.image = frame.data;

        for (int i = 0; i < face_results.face_location.size(); i++)
        {
            detect_result box = face_results.face_location[i];
            point cur_points[5];

            cur_points[0].x = box.landmark[0];
            cur_points[0].y = box.landmark[1];
            cur_points[1].x = box.landmark[2];
            cur_points[1].y = box.landmark[3];
            cur_points[2].x = box.landmark[4];
            cur_points[2].y = box.landmark[5];
            cur_points[3].x = box.landmark[6];
            cur_points[3].y = box.landmark[7];
            cur_points[4].x = box.landmark[8];
            cur_points[4].y = box.landmark[9];

            get_roi(cur_points, input_imgs[0], tmp_input_img, box);
            //get_inv_affine_matrix(cur_points, target_points, affine_matrix);
            //inv_warp_affine(tmp_input_img, input_imgs[0], affine_matrix);
            Mat tmp(input_imgs[0].image_attr.u32Width, input_imgs[0].image_attr.u32Height, CV_8UC3, input_imgs[0].image);
            imshow("tmp", tmp);
            p_rec_machine->nce_alg_inference(input_imgs);
            p_rec_machine->nce_alg_get_result(tmp_results);
            cur_id = *(FaceID*)tmp_results.st_alg_results->obj;

            float sim = getSimilarity_c(label_id.face_id, cur_id.face_id, label_id.dims);
            face_results.face_name[i].cos_sim = sim;
            if (sim > sim_thresh)
            {
                face_results.face_name[i].name = name;
            }
            else
            {
                face_results.face_name[i].name = "Unknown";
            }
        }
        return NCE_SUCCESS;
    }
private:
    char name[20];
    float affine_matrix[9];
    Mat label_img;
    Mat cur_img;

    point target_points[5];
    unique_ptr<nce_alg_machine> p_rec_machine;
    FaceID label_id;
    FaceID cur_id;
    vector<img_info> input_infos;
    vector<img_t> input_imgs;
    alg_result_info tmp_results;

    float getMold_c(float* vec, NCE_S32 dims)
    {
        int   n = dims;
        float sum = 0.0;

        for (int i = 0; i < n; ++i)
        {
            sum += vec[i] * vec[i];
        }

        return sqrt(sum);
    }

    float getSimilarity_c(float* lhs, float* rhs, NCE_S32 dims)
    {
        int   n = dims;
        float tmp = 0.0;
        for (int i = 0; i < n; ++i)
            tmp += lhs[i] * rhs[i];

        return tmp / (getMold_c(lhs, dims) * getMold_c(rhs, dims));
    }

    int get_roi(point cur_points[5], img_t& tmp_img, img_t & tmp_img_ori, detect_result roi_bbox)
    {
        //Point2f src[4];
        //Point2f dst[4];

        //src[0].x = cur_points[3].x;
        //src[0].y = cur_points[3].y;
        //src[1].x = cur_points[4].x;
        //src[1].y = cur_points[4].y;
        ////src[2].x = cur_points[2].x;
        ////src[2].y = cur_points[2].y;
        //src[2].x = cur_points[0].x;
        //src[2].y = cur_points[0].y;
        //src[3].x = cur_points[1].x;
        //src[3].y = cur_points[1].y;

        //dst[0].x = target_points[3].x;
        //dst[0].y = target_points[3].y;
        //dst[1].x = target_points[4].x;
        //dst[1].y = target_points[4].y;
        ////dst[2].x = target_points[2].x;
        ////dst[2].y = target_points[2].y;
        //dst[2].x = target_points[0].x;
        //dst[2].y = target_points[0].y;
        //dst[3].x = target_points[1].x;
        //dst[3].y = target_points[1].y;

        //Mat M = getAffineTransform(src, dst);

        //Mat src_img(tmp_img_ori.image_attr.u32Height, tmp_img_ori.image_attr.u32Width, CV_8UC3, tmp_img_ori.image);
        //Mat dst_img;
        //for (int i = 0; i < 5; i++)
        //{
        //    circle(src_img, src[i], 10, Scalar(0, 0, 255));
        //}
        //warpAffine(src_img, dst_img, M, Size(112, 112));
        //memcpy(tmp_img.image, dst_img.data, dst_img.cols * dst_img.rows * 3 * sizeof(NCE_U8));

        get_inv_affine_matrix(cur_points, target_points, affine_matrix);
        inv_warp_affine(tmp_img_ori, tmp_img, affine_matrix);

        //Mat ori_img(tmp_img_ori.image_attr.u32Height, tmp_img_ori.image_attr.u32Width, CV_8UC3, tmp_img_ori.image);
        //Mat roi(ori_img, Rect(roi_bbox.x1, roi_bbox.y1, roi_bbox.x2 - roi_bbox.x1, roi_bbox.y2 - roi_bbox.y1));
        //resize(roi, roi, Size(tmp_img.image_attr.u32Width, tmp_img.image_attr.u32Height));
        //memcpy(tmp_img.image, roi.data, roi.cols * roi.rows * 3 * sizeof(NCE_U8));
        return 0;
    }

};

int main(int argc, char* argv[])
{
    char* retinaface_path = argv[1];
    char* arcface_path = argv[2];
    char* label_img = argv[3];
    char* tddfa_path = argv[4];
    OSA_DEBUG_DEFINE_TIME;
    VideoCapture cap(0);
    Mat frame;


    try
    {
        FaceDetect face_detect(retinaface_path, tddfa_path);
        FaceRec    face_rec(arcface_path, label_img, face_detect);
        while (1)
        {
            if (!cap.read(frame))
            {
                printf("open camera failed! please check your camera");
            };
            FaceResults face_results;
            OSA_DEBUG_START_TIME
            face_detect.get_face(frame, face_results);
            face_rec.compare_face(frame, face_results);
            OSA_DEBUG_END_TIME(infer cost:)
            draw_bboxes(frame, face_results);
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