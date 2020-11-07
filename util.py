import numpy as np 
import os 
import cv2 
import matplotlib.pyplot as plt 
import csv 
import json
from collections import Counter 
from matplotlib.patches import Rectangle
import math
import tensorflow as tf 
keras = tf.keras 

# DOCUMENTATION 


############ CLOUD USAGE  (or local if you want to load a fraction of the dataset (see f param below) )

# LOAD DATASET

# !! TLDR; RUN THIS ! --->  x_train,y_train,x_val,y_val,x_test,y_test = data_load(f=1)

    # f param below is the fraction of the dataset you want to load
    # for example, pass in f=0.1 to load only 10% of the train,val,and test sets
    # this is useful if you want to test locally and load only a subset of data to fit in RAM
    # also note that the png files are COMPRESSED, and thus the RAM memory footprint of the png files
    # is approximately 2 to 7 times larger (uncompressed) 
    # see the data_load() implementation directly below for more information
    # note: the output will say "Removed x lesions.." this is because the neighboring slice was missing for those slices 

def data_load(dl_info, dl_info_vector, json_labels, organ_id, f=1)  : 
    
    organ_data = load_all_data_for_term(dl_info, dl_info_vector, json_labels, organ_id,f) #8 is the term ID which corresponds to liver lesion 

    # the above is defined below and returns a dict with keys train, val, test , each of which has value that is a tuple of [ X,Y ]
    # so we can destructure this like so:
    from operator import itemgetter
    [x_train,y_train] , [x_val, y_val] , [x_test,y_test] = itemgetter("train","val","test")(organ_data)
    # and then we can return all these elements in one big tuple
    
    print("Train Size: {}\nVal Size: {}\nTest Size: {}".format(str(len(x_train)),str(len(x_val)),str(len(x_test)))) 
    return (x_train,y_train,x_val,y_val,x_test,y_test)



# END LOAD DATASET

############ END CLOUD USAGE 


# END DOCUMENTATION 


# manage windows/linux stuff
import platform 
_os = platform.system() 
if _os == "Linux"  : 
    fdelim = "/"
elif _os == "Darwin" : 
    fdelim = "/"
elif _os == "Windows" :     
    fdelim  = "\\" 
else : 
    print("unrecognized os!") 


def set_image_dir(d) : 
    global image_dir
    global dl_info_vector
    global dl_info 
    
    image_dir = d 
    dl_info_vector = read_dl_info_vector() 
    dl_info = read_dl_info() 
    

def get_files() : 
    # prepare the image directories 
    image_dir = "images" + fdelim + "Images_png" + fdelim 
    sub_dirs = os.listdir(image_dir) 
    sub_dirs.sort() 

    # replace each sub_dir with [ sub_dir [file_list] ] 
    files = [] 
    for d in sub_dirs : 
        sub_files = os.listdir(os.path.join(image_dir,d))
        sub_files.sort() 
        sub_files_fp = [ os.path.join(image_dir,d,x) for x in sub_files ] 
        files.append( [ d , sub_files_fp ] ) 
    
    return files 


# helper functions

def check_for_file(fname)  :   
    
    import os.path
    return os.path.isfile(fname) 


def append_file(fname, strang) : 
    if not check_for_file(fname) : 
        mode = 'w' 
    else : 
        mode = 'a+' 

    with open(fname, mode) as outfile : 
        outfile.write(strang)

# given a file can we produce a numpy array 
def read_image(dl_info, fn,with_win=False,bb=True,verbose=True) : 
    im = cv2.imread(fn,-1) 
    im =  (im.astype(np.int32)-32768).astype(np.int16) 
    
    # only look up the window if with_win is False 
    win = with_win or [float(x) for x in dl_info[fn]['DICOM_windows'].split(",")]

    if verbose : 
        print("For fn: {}, using window: [{},{}]".format(fn,win[0],win[1]))
        
    im = windowing(im,win) 
        
    # now get the bounding box as well 
    if bb :
        _bb = [round(float(x)) for x in dl_info[fn]['Bounding_boxes'].split(',') ] 
        return (im , _bb, win )
    else : 
        return im 


def gen_neighbor_names(fn) : 

    tok = fn.split(fdelim) # ['images', 'Images_png', '000001_01_01', '103.png']
    
    slice_tok = tok[-1].split(".")
    left_num  = "{:03d}".format(int(slice_tok[0]) - 1)
    right_num = "{:03d}".format(int(slice_tok[0]) + 1)
    left_fn   = fdelim.join(tok[0:-1]) + fdelim + left_num + ".png" 
    right_fn   = fdelim.join(tok[0:-1]) + fdelim + right_num + ".png" 
    return (left_fn, right_fn) 


def read_image_and_neighbors(dl_info, fn,verbose=True) : 
    # should be able to assume that the slices are available on either side 
    lfn, rfn = gen_neighbor_names(fn) 
    
    # first we read the main image and get the window and bounding box 
    mim, bb, win = read_image(dl_info, fn,verbose=verbose) 
    
    # now we will read the left and right images using the same window and w/o bb
    lim = read_image(dl_info, lfn,with_win=win,bb=False,verbose=verbose)
    rim = read_image(dl_info, rfn,with_win=win,bb=False,verbose=verbose)

    # are going to produce a matrix (512,512,3) 
    slices = np.zeros( (512,512,3 ) ) 

    try:
        slices[:,:,0] = lim
    except:
        pass
    
    try:
        slices[:,:,1] = mim 
    except:
        pass
    try:
        slices[:,:,2] = rim 
    except:
        pass
    
    return (slices, np.array(bb)) 

def nb_imshow(im,bb=False) : 
  
    plt.imshow(im[:,:,1],cmap='gray')

    # if bounding box will also draw the bb 
    if bb.any() : 
        # unnormalize the bounding box 
        bb = 512*bb 
        # need to convert to appropriate shapes 
        pt = (bb[0], bb[1])
        w  = bb[2] - bb[0]
        h  = bb[3] - bb[1]
        print("Using bb coords: ({},{}),{},{}".format(pt[0],pt[1],w,h))
        plt.gca().add_patch(Rectangle(pt,w,h,linewidth=1,edgecolor='lime',facecolor='none'))

def show_image(im,bb=False) : 
    plt.gca().cla()
    plt.imshow(im,cmap='gray')
    plt.ion()
    plt.show() 
    
    # if bounding box will also draw the bb 
    if bb.any() : 
        # need to convert to appropriate shapes 
        pt = (bb[0], bb[1])
        w  = bb[2] - bb[0]
        h  = bb[3] - bb[1]
        print("Using bb coords: ({},{}),{},{}".format(pt[0],pt[1],w,h))
        plt.gca().add_patch(Rectangle(pt,w,h,linewidth=1,edgecolor='lime',facecolor='none'))

    plt.draw()
    plt.pause(0.001) # non blocking 
        
def disp(fn,bb=False) : 
    im, bb, win = read_image(dl_info, fn)  # read the image and the bounding box 
    if bb :
        show_image(im,bb=bb) 
    else : 
        show_image(im) 

def disp_loop() :
    plt.figure()
    for folder in files  :
        for f in folder[1] :
            im = read_image(f)
            plt.imshow(im,cmap='gray')
            plt.pause(0.1)
            plt.draw()

def test_show() :
    disp("images/Images_png/000001_03_01/088.png", bb=True)
    
def read_json_labels(path_to_json_labels = 'text_mined_labels_171_and_split.json') :             
    with open(path_to_json_labels) as json_file: 
        data = json.load(json_file)
        return data 
    
def get_index_of_term(t) : 
    return json_labels['term_list'].index(t)

def search_for_term(term, to_search) :   # term is actually an index here  
    matches = [] 
    for i,val in enumerate(to_search) : 
        # each val here is a list [x, x2, x3.. ] 
        # if 'term' is in this list then we add it to matches 
        if term in val : 
            matches.append([i,val])
    return matches
    
    
def read_dl_info_vector(image_dir = "images" + fdelim + "Images_png" + fdelim  , DL_INFO_PATH = './') : 
    
    #function for modifying map object after generated 
    def transform_map(m) : 
        #fixes fname 
        tok = m['File_name'].split("_")  
        m['File_name'] = image_dir +  "_".join(tok[0:3]) + fdelim + tok[-1]            
        return m 
        
    with open(DL_INFO_PATH + 'DL_info.csv') as f:
        a = [{k: v for k, v in row.items()}
             for row in csv.DictReader(f, skipinitialspace=True)]
        return [transform_map(x) for x in a] 

def read_dl_info() : 
    info = {} 
    a  = read_dl_info_vector(
    image_dir = '../images/Images_png/',
    DL_INFO_PATH  = '../cs230/') 
    for d in a : 
        info[d['File_name']] = d 
    return info 

def select_lesion_idxs(dl_info_vector, s) : 
    
    return [ dl_info_vector[x] for x in s ] 


def get_folders_for_lesions_set(ls) :
    return [ "/".join(x['File_name'].split("/")[0:3]) for x in ls ]

def fname_with_neighbors(fname) :
    (ln, rn) = gen_neighbor_names(fname)
    return [ln, fname, rn ]

def get_fnames_and_neighbors_for_lesions_set(ls) :
    res =  [ fname_with_neighbors(x['File_name'])  for x in ls  ]
    return [item for sublist in res for item in sublist if check_for_file(item) ] 

def write_list_to_file(fname,l) :
    for i in l :
        append_file(fname,i + "\n")
        

def generate_term_specific_set(dl_info_vector, json_labels, train_val_test, term,v=True) : 
    labs = search_for_term(term, json_labels['{}_relevant_labels'.format(train_val_test)])

    labs_idx   = [ x[0] for x in labs ]

    lesion_idx = [ json_labels['{}_lesion_idxs'.format(train_val_test)][i] for i in labs_idx  ] 

    lesions    = select_lesion_idxs(dl_info_vector, lesion_idx) 

    coarse_types = Counter([x['Coarse_lesion_type'] for x in lesions])

    def filt(l) :
        ln,rn = gen_neighbor_names(l['File_name'])
        #print(ln) 
        return (check_for_file(ln) and check_for_file(rn) )
    
    final_lesions = list(filter( filt , lesions))
    
    if v : 
        print("Removed {} lesion(s) of {}".format(len(lesions) - len(final_lesions) , len(lesions)))

    return { "lesions" : final_lesions , 
             "coarse_types" : coarse_types , 
             "lesion_idx"  : lesion_idx , 
             "labs_idx" : labs_idx , 
             "labs" : labs } 



def load_data_to_memory(dl_info, lesions,msg=None) : 
    num_lesions  = len(lesions) 
    xs = np.zeros( (num_lesions, 512,512,3 ) ) 
    ys = np.zeros( (num_lesions, 4 ) ) 
    
    if msg :
        print(msg)
    
    for i,v in enumerate(lesions) :

        if (i % 100 == 0 and i != 0 ) : 
            print("On index: " + str(i))
        
        # get the filename of the lesion 
        fn = lesions[i]['File_name']
        
        # get the data 
        slices,bounding_box = read_image_and_neighbors(dl_info, fn,verbose=False) 
        
        # append the data
        xs[i,:,:,:] = slices 
        ys[i,:]   = bounding_box/512

    # now we return the xs and ys 
    return (xs,ys) 

def load_all_data_for_term(dl_info, dl_info_vector, json_labels, t, f=1) :
    sets = ["train" , "val" , "test" ]
    
    print("\nLoading data for term index: " + str(t) )
    print("Fraction of data that will be loaded={}\n".format(f)) 
    data = {} 
    for s in sets :
        print("Loading {} set".format(s))        
        term_dataset_with_metadata = generate_term_specific_set(dl_info_vector, json_labels, s,t)
        max_index = int(len(term_dataset_with_metadata["lesions"])*f)
        data[s] = load_data_to_memory(dl_info, term_dataset_with_metadata["lesions"][0:max_index])
        print("Done\n")
        
    return  data 




def build_partitioned_dataset(lesions,name,num_parts) : 

    num_per_part = math.floor(len(lesions)/num_parts) 
    print("{} per part".format(num_per_part))
    
    for k in range(0,num_parts) : 
        part = lesions[num_per_part*k:num_per_part*(k+1)]
        part_number = str(k+ 1)
        print("Building part #{} with {} lesions".format(part_number,len(part)))
        build_dataset(part,name+"_part_"  + part_number)
        
    if len(lesions) % num_per_part != 0 : 
        part = lesions[num_per_part*num_parts:len(lesions)]
        part_number = str(k+ 1)
        print("Building part #{} with {} lesions".format(part_number,len(part)))
        build_dataset(part,name+"_part_"  + part_number)
    
    #done 

    
    

def build_dataset(lesions,name) : 
    num_lesions  = len(lesions) 
    xs = np.zeros( (num_lesions, 512,512,3 ) ) 
    ys = np.zeros( (num_lesions, 1, 4 ) ) 
    
    print("Generating data set...") 
    
    for i,v in enumerate(lesions) : 
        
        #if ( i % 200 == 0 ) : 
        #  print("On index: " + str(i)) 
        
        # get the filename of the lesion 
        fn = lesions[i]['File_name']
        
        # TODO get the data -- WILL wrap INSIDE TRY CATCH and if error then will 
        # print the FILENAME so I can explore where the error happened
        slices,bounding_box = read_image_and_neighbors(dl_info, fn,verbose=False) 
        
        # append the data
        xs[i,:,:,:] = slices 
        ys[i,:,:]   = bounding_box 
        
    # at this point data set should be built 
    # will write the data to numpy binary file 
    #print("Saving xs...")
    np.save(name + '_xs',xs)
    #print("Saving ys...")
    np.save(name + '_ys',ys)
    print("Done!") 

    

def get_dataset() : 
    # FOR LOADING XS and YS
    fbase = "datasets" + fdelim + "liver_train_part_1_"
    xs_fn = fbase + "xs.npy"
    ys_fn = fbase + "ys.npy"    
    xs = np.load(xs_fn)
    ys = np.load(ys_fn)
    return (xs, ys ) 
    
def plot_data(xs,ys,num) : 
    show_image(xs[num,:,:,1] , bb=ys[num,:,:].flatten() )

def windowing2(im,win): 
    im = im.astype(float)
    return np.min(255, np.max(0, 255*(im-win[0])/(win[1]-win[0])))

def windowing(im, win):
    # (https://github.com/rsummers11/CADLab/blob/master/LesaNet/load_ct_img.py) 
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= (win[1] - win[0])
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1    


def reload() : 
    import importlib 
    import sys
    importlib.reload(sys.modules['util'])

 
def convert_to_iou_format(y) : 
    """  
    Will convert from [x_min, y_min, x_max, y_max] to [x, y, width, height] 
    """
    return np.array([ y[0] , y[1] , y[2]-y[0] , y[3]-y[1] ] ) 


def calculate_iou(y_true, y_pred):
    
    """
    Input:
    Keras provides the input as numpy arrays with shape (batch_size, num_columns).
    
    Arguments:
    y_true -- first box, numpy array with format [x, y, width, height, conf_score]
    y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
    x any y are the coordinates of the top left corner of each box.
    
    Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)
    
    """

    results = []
    
    
    # set the types so we are sure what type we are using
    #y_true = convert_to_iou_format(y_true.astype(np.float32))
    
    #y_pred = convert_to_iou_format(y_pred.astype(np.float32))   


    # boxTrue
    x_boxTrue_tleft = y_true[0]  # numpy index selection
    y_boxTrue_tleft = y_true[1]
    boxTrue_width = y_true[2]
    boxTrue_height = y_true[3]
    area_boxTrue = (boxTrue_width * boxTrue_height)

        # boxPred
    x_boxPred_tleft = y_pred[0]
    y_boxPred_tleft = y_pred[1]
    boxPred_width = y_pred[2]
    boxPred_height = y_pred[3]
    area_boxPred = (boxPred_width * boxPred_height)


    x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
    y_boxTrue_br = y_boxTrue_tleft + boxTrue_height # Version 2 revision
	# calculate the top left and bottom right coordinates for the intersection box, boxInt

    x_boxPred_br = x_boxPred_tleft + boxPred_width
    y_boxPred_br = y_boxPred_tleft + boxPred_height


	# boxInt - top left coords
    x_boxInt_tleft = np.max([x_boxTrue_tleft,x_boxPred_tleft])
    y_boxInt_tleft = np.max([y_boxTrue_tleft,y_boxPred_tleft]) # Version 2 revision

	# boxInt - bottom right coords
    x_boxInt_br = np.min([x_boxTrue_br,x_boxPred_br])
    y_boxInt_br = np.min([y_boxTrue_br,y_boxPred_br]) 

	# Calculate the area of boxInt, i.e. the area of the intersection 
	# between boxTrue and boxPred.
	# The np.max() function forces the intersection area to 0 if the boxes don't overlap.
	
	
	# Version 2 revision
    area_of_intersection = \
    np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])

    iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)


	# This must match the type used in py_func
    iou = iou.astype(np.float32)
	
    
    # return the mean IoU score for the batch
    return iou 

 
def IoU(y_true, y_pred):
    
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours 
    # trying to debug and almost give up.
    
    iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)

    return iou 

    
if __name__ == '__main__' :
    
    pass
