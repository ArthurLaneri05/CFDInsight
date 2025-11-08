### Pvserver settings
pvserver_port = 11111
pvserver_np = 10        # set up according to your pc specs


### Path to ParaView modules
paraview_path = "/.../site-packages"


### List of ParaView modules to load
additional_modules = []


### Output dict name
output_folder_title = "CFDInsight_output"


### Slices global settings
camera_distAlongNormal= 100  # in meters


### Raw image global settings
img_size_in = (9, 6)
raw_img_compression_level = 4   # value in (0 [No compression], 9 [Max compression])
img_dpi = 300
background_color = [.9, .9, .9]


### Refined image global settings
pad_in = .1
offset_label_spacing_in = .1

title_font_size = 10
title_linespacing = 1.7
label_font_size = 8
tick_font_size = 8

# Using formula to estimate the maximum number of characters per line 
# alpha is the average character width as a fraction of font size (em units)
alpha = 0.8