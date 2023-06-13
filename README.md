# Partial_PCDs



How to generate the partial PCDs from complete.


You should have the following files.folders in the currect directory
	1. partial_pcd.py
	2. cameras.npz
	3. dataset       (dataset/models/02691156/"object_id".ply) 
			 where, "object_id" the id of the object





Step 1: 
======
Run generate_img_depth() from the file "partial_pcd.py"  [python partial_pcd.py]
output:	It will generate a folder 'partialPCD' that contains RGB and depth images in 24 different poses
	images: (partialPCD/02691156/"object_id"/depth/**.png) 	



Step 1: 
======
Run generate_partial_pcds() from the file "partial_pcd.py"  [python partial_pcd.py]
process:	It will read the RGB and depth images one by one, generates the partial pcd and save in the output directory.	
output:		The generated PCDs will be saved in the same directory. See the next two lines;
		1. generic pose PCDs: partialPCD/02691156/"object_id"/pcds_24poses/**.xyz
		1. canonical pose PCDs: partialPCD/02691156/"object_id"/pcds_canonical/**.xyz
