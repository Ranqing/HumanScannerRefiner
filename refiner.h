#ifndef REFINER_H
#define REFINER_H

#include "../Qing/qing_common.h"

class StereoRefiner
{
public:
    StereoRefiner(string frame_folder, string mask_folder, string stereo_folder, string scanner_folder, string frame_name, string stereo_fn);

    //   vector<unsigned char> m_view_l, m_view_r, m_mask_l, m_mask_r;
    //   vector<unsigned char> m_gray_l, m_gray_r, m_census_l, m_census_r;
    vector<float> m_disp_l, m_disp_r;
    vector<float> m_mcost_vol_l, m_mcost_vol_r;
    vector<unsigned char> m_census_l, m_census_r;

    Mat m_raw_view_l, m_raw_view_r, m_raw_mask_l, m_raw_mask_r;
    Mat m_view_l, m_view_r, m_mask_l, m_mask_r;
    Mat m_gray_l, m_gray_r;
    Mat m_refine_disp, m_median_refine_disp;                       //CV_32FC1
    Mat m_show_disp, m_median_show_disp;                         //CV_16UC1

    int m_stereo_id;
    float m_min_disp, m_max_disp;           //start disparity  ~  end disparity
    int m_d_range;                          //disparity range:  (m_max_disp - m_min_disp) / m_disp_step + 1

    string m_scanner_dir;                   //directory of scanner results
    string m_frame_name, m_stereo_name;
    Point2i m_crop_point_l, m_crop_point_r;
    int m_crop_w, m_crop_h, m_crop_size;
    Mat m_qmatrix;

    //params
    float m_disp_step;
    int m_wnd;
    int m_patch_size;              //divide images in y-direction
    int m_patch_overlap;           //overlap

    string m_out_dir;

    void init_params();
    void read_in_scanner_results(const int level);
    void sgbm_refine(const int level);
    void triangulate(const int level);

    void qing_call_qx_rbf(Mat& disp);

    //	void census_transform();
    //  void cal_census_mcost_vol(vector<unsigned char>& cost_vol, vector<unsigned char>& census_l, vector<unsigned char>& census_r, int start_y,  int end_y, int direction);
};

#endif // REFINER_H
