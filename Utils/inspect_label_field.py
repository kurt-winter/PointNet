from plyfile import PlyData

#ply_path = "/data/AI_Dev_Data/Paris-Lille-3D/training_10_classes/Lille2.ply"
ply_path = "/data/AI_Dev_Data/Paris-Lille-3D/test_10_classes/ajaccio_2.ply"
plydata = PlyData.read(ply_path)

print("Available fields:")
print(plydata['vertex'].data.dtype.names)