#include "refiner.h"

int main(int argc, char * argv[])
{
    string img_folder  = "/media/ranqing/Work/ZJU/HumanDatas_20161224/Humans_frame/";
    string msk_folder  = "/media/ranqing/Work/ZJU/HumanDatas_20161224/Humans_mask/" ;
    string stereo_folder = "/media/ranqing/Work/ZJU/HumanDatas_20161224/Humans_stereo/";
    string result_folder = "/media/ranqing/Work/ZJU/HumanDatas_20161224/Humans_result/";

    cout << "usage: " << argv[0] << " FRM_0245 stereo_A01A02.info" << endl;           //stereo

    if(argc != 3)
    {
        cerr << "invalid arguments.." << endl;
        return -1;
    }
    string frame_name = argv[1];
    string stereo_fn  = argv[2];

    StereoRefiner * refiner = new StereoRefiner(img_folder, msk_folder, stereo_folder, result_folder, frame_name, stereo_fn);	
	refiner->init_params();

    int st = 0;
    for(int level = st; level >= 0; level--)
	{
        refiner->sgbm_refine(level);
		refiner->triangulate(level);
	}

//	refiner->census_transform();
//	refiner->sgbm_refine();
//	refiner->triangulate();

    return 1;
}
