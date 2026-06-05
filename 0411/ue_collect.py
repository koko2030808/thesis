import unreal, json, math

skel_comp = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()[0].get_component_by_class(unreal.SkeletalMeshComponent.static_class())
skel_comp.set_position(0.0, False)
unreal.EditorLevelLibrary.editor_invalidate_viewports()

sockets = [
    "H36M_00_Pelvis", "H36M_01_R_Hip", "H36M_02_R_Knee", "H36M_03_R_Ankle", 
    "H36M_04_L_Hip", "H36M_05_L_Knee", "H36M_06_L_Ankle", 
    "H36M_07_Spine", "H36M_08_Thorax", "H36M_09_Nose", "H36M_10_Head", 
    "H36M_11_L_Shoulder", "H36M_12_L_Elbow", "H36M_13_L_Wrist", 
    "H36M_14_R_Shoulder", "H36M_15_R_Elbow", "H36M_16_R_Wrist"
]

data = [[[skel_comp.get_socket_location(s).x, skel_comp.get_socket_location(s).y, skel_comp.get_socket_location(s).z] for s in sockets]]

with open("C:/Users/Public/H36M_Sequence_10Frames.json", 'w') as f: json.dump(data, f)
unreal.log("【單幀重置】已收割最新藍圖縮放數據！")