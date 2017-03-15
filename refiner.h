#ifndef REFINER_H
#define REFINER_H

#include "../../Qing/qing_common.h"

class StereoRefiner
{
public:
    StereoRefiner(string frame_folder, string mask_folder, string stereo_folder, string result_folder, string frame_name, string stereo_fn );

    vector<unsigned char> m_view_l, m_view_r, m_mask_l, m_mask_r;
    vector<unsigned char> m_gray_l, m_gray_r, m_census_l, m_census_r;
    vector<float> m_disp_l, m_disp_r;
    vector<float> m_mcost_vol_l, m_mcost_vol_r;

    //test sgbm
    Mat m_view_mat_l, m_view_mat_r, m_mask_mat_l, m_mask_mat_r, m_view_disp;
    //end of test sgbm

    int m_stereo_id;
    float m_min_disp, m_max_disp;          //start disparity  ~  end disparity
    int m_d_range;                         //disparity range:  (m_max_disp - m_min_disp) / m_disp_step + 1

    string m_result_dir;                    //directory of scanner results
    string m_frame_name, m_stereo_name, m_cam_name_l, m_cam_name_r;
    Mat m_raw_view_l, m_raw_view_r, m_raw_mask_l, m_raw_mask_r;
    Point2i m_crop_point_l, m_crop_point_r;
    int m_crop_w, m_crop_h, m_crop_size;
    Mat m_qmatrix;


    //params
    float m_disp_step;
    int m_wnd;
    int m_patch_size;              //divide images in y-direction
    int m_patch_overlap;           //overlap

    string m_out_dir;

    void read_in_results();
    void init_params();
    void census_transform();
    void cal_census_mcost_vol(vector<unsigned char>& cost_vol, vector<unsigned char>& census_l, vector<unsigned char>& census_r, int start_y,  int end_y, int direction);
    void refine();

    void triangulate();



};

#endif // REFINER_H
