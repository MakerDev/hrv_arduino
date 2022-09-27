import csv
import os

def extract_motion_only(root):
    actors_path = f"{root}/BVH"
    export_folder = f"{root}/BVH_only_motion"

    if not os.path.exists(export_folder):
        os.mkdir(export_folder)

    actors = os.listdir(actors_path)
    for actor in actors:
        actor_folder = os.path.join(actors_path, actor)
        bvh_files = os.listdir(actor_folder)
        export_actor_folder = os.path.join(export_folder, actor)
        
        if not os.path.exists(export_actor_folder):
            os.mkdir(export_actor_folder)

        for bvh_file in bvh_files:
            bvh_path = os.path.join(actor_folder, bvh_file)
            with open(bvh_path) as f:
                motions = f.readlines()[350:]
                export_file_path = os.path.join(export_actor_folder, bvh_file)
                with open(export_file_path, 'w', newline='') as f_out:
                    f_out.writelines(motions)

if __name__ == "__main__":
    root = "D:/AESPA Dataset/kinematic-dataset"
    #1. file_info.csv로부터, 각 파일의 이름을 키로 하여, 그 파일의 정보를 담고있는 dict를 생성한다.
    file_info_path = f"{root}/file-info.csv"
    file_infos = {}
    with open(file_info_path) as f:
        reader = csv.reader(f)
        
        header = next(reader)

        for line in reader:
            filename = line[0]
            file_info = {}
            for i, info in enumerate(line[1:]):
                file_info[header[i+1]] = info
            file_infos[filename] = file_info

    #2. export motion rows only.
    #extract_motion_only(root)

    #3. featrue extraction
    #각 열 별로 평균, 표준편차 등등을 구해서 
    