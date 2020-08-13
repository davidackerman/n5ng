#!/usr/bin/env python3

import argparse
import gzip
import io
import zarr
import numpy as np
from flask import Flask, jsonify, Response, request, redirect
from flask_cors import CORS
import os, glob
import csv
import hashlib
import re
from math import log10, floor

def round_sig(x, sig=2):
   return round(x, sig-int(floor(log10(abs(x))))-1)

app = Flask(__name__)
CORS(app)
global source_n5

def convertStringToInt(s):
    m = hashlib.md5()
    m.update(s.encode('utf8'))
    print(str(int(m.hexdigest(), 16)))
    return int(str(int(m.hexdigest(), 16))[-19:])

def get_scales(dataset_name, scales=[], encoding='raw', base_res=np.array([1.0,1.0,1.0])):
    dataset_name = re.sub('_n5ngSetValue\d*','',dataset_name)

    def get_scale_for_dataset(dataset, scale, base_res):
        if 'resolution' in dataset.attrs:
            resolution = dataset.attrs['resolution']
        elif 'pixelResolution' in dataset.attrs:
            resolution = dataset.attrs['pixelResolution']['dimensions']
        elif 'downsamplingFactors' in dataset.attrs:
            # The FAFB n5 stack reports downsampling, not absolute resolution
            resolution = (base_res * np.asarray(dataset.attrs['downsamplingFactors'])).tolist()
        else:
            resolution = (base_res*2**scale).tolist()
        return {
                    'chunk_sizes': [list(reversed(dataset.chunks))],
                    'resolution': resolution,
                    'size': list(reversed(dataset.shape)),
                    'key': str(scale),# if hasKey,
                    'encoding': encoding,
                    'voxel_offset': [x*1.0/resolution[0] for x in dataset.attrs.get('offset', [0,0,0])],
                }

    if  scales:
        # Assumed scale pyramid with the convention dataset/sN, where N is the scale level
        scale_info = []
        for scale in scales:
            try:
                #print("helloooow %s/s%d" % (dataset_name, scale))
                dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
                dataset = app.config['n5file'][dataset_name]#_with_scale]
                this_scale = scale_info.append(get_scale_for_dataset(dataset, scale, base_res))
            except Exception as exc:
                print(exc)
    else:
        dataset = app.config['n5file'][dataset_name]
        # No scale pyramid for this dataset
        scale_info = [ get_scale_for_dataset(dataset, 1.0, base_res) ]
        #print(scale_info)
    return scale_info

@app.route('/<path:dataset_name>/info')
def dataset_info(dataset_name):
    dataset_name = re.sub('_n5ngSetValue\d*','',dataset_name)

    if "mesh" in dataset_name:
        info = {"@type": "neuroglancer_legacy_mesh"
                #"@type": "neuroglancer_multilod_draco",
                #"vertex_quantization_bits": 16,
                #"lod_scale_multiplier": 1
        }
    elif "_properties" in dataset_name:
        dataset = dataset_name.split("_properties")[0]

        mesh_ids = []
        volume_strings = []
        data_path = source_n5+"/"+dataset+"/mesh/data.csv"
        if os.path.exists(data_path):
            mesh_ids = []
            volumes = []
            with open(data_path) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                next(readCSV) #skip header
                for row in readCSV:
                    mesh_ids.append(row[0])
                    volume = float(row[1])
                    volumes.append(volume)

                    volume = round_sig(volume,4)
                    x = round(float(row[3])/4)
                    y = round(float(row[4])/4)
                    z = round(float(row[5])/4)
                    volume_strings.append(f"Volume (nm^3): {volume}, COM (nm)({x},{y},{z})")

            mesh_ids = [mesh_id for _,mesh_id in sorted(zip(volumes,mesh_ids), reverse=True)]
            zero_padding = len(str(len(mesh_ids)))
            idx = 0
            for _,volume_string in sorted(zip(volumes,volume_strings), reverse=True):
                volume_strings[idx] = f"(idx: {str(idx).zfill(zero_padding)}) {volume_string}"
                idx+=1
            #volume_strings = [volume_string ]
        else:
            mesh_ids = [x.split("/")[-1] for x in glob.glob(source_n5+"/"+dataset+"/mesh/*.ngmesh")]
            mesh_ids = [mesh.split(".")[0] for mesh in mesh_ids]
            volume_strings = [""]*len(mesh_ids)
        #print(volume_dictionary)         
        #print(volume_strings)

        info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
        "ids": mesh_ids,
        "properties":[
            {"id": "label",
            "type": "label",
            "values": volume_strings
            },
            {
            "id": "description",
            "type": "description",
            "values": volume_strings
            }
            ]
        }
        }
    else:# elif "s0" in dataset_name or "s1" in dataset_name or "s2" in dataset_name:

      info = {
            'data_type' : 'uint8' if ( ("medialSurface" in dataset_name ) or ("binarized" in dataset_name)) else 'uint64',
            'type': 'segmentation',
            'num_channels' : 1,
            'mesh' : 'mesh',
            'scales' : get_scales(dataset_name, scales=list(range(0,1)), base_res=np.array([2.0, 2.0, 2.0])) if ("training" in dataset_name) else get_scales(dataset_name, scales=list(range(0,1)))#, base_res=np.array([4.0, 4.0, 4.0])) # get_scales(dataset_name, scales=list(range(0,8)), base_res=np.array([4.0, 4.0, 40.0]))
        }
    #print(info)
    return jsonify(info)


# Implement the neuroglancer precomputed filename structure for URL requests
@app.route('/<path:dataset_name>/<int:scale>/<int:x1>-<int:x2>_<int:y1>-<int:y2>_<int:z1>-<int:z2>')
def get_data(dataset_name, scale, x1, x2, y1, y2, z1, z2):

    setValue = False
    if("_n5ngSetValue" in dataset_name):
        setValue = int(dataset_name.split('_n5ngSetValue')[-1])
        dataset_name = re.sub('_n5ngSetValue\d*','',dataset_name)

    # TODO: Enforce a data size limit
    dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
    dataset = app.config['n5file'][dataset_name]#_with_scale]
    resolution = dataset.attrs['pixelResolution']['dimensions']

    voxel_offset = [int(x*1.0/resolution[0]) for x in dataset.attrs.get('offset', [0,0,0])]
    #print("heasdfafadsfaf", voxel_offset)
    x1-=voxel_offset[0]
    x2-=voxel_offset[0]
    y1-=voxel_offset[1]
    y2-=voxel_offset[1]
    z1-=voxel_offset[2]
    z2-=voxel_offset[2]
    #print(x1,x2,y1,y2,z1,z2)
    data = dataset[z1:z2,y1:y2,x1:x2]
    if setValue:
        data[data>0] =  setValue
    # Neuroglancer expects an x,y,z array in Fortram order (e.g., z,y,x in C =)
    response = Response(data.tobytes(order='C'), mimetype='application/octet-stream')

    accept_encoding = request.headers.get('Accept-Encoding', '')
    if 'gzip' not in accept_encoding.lower() or \
           'Content-Encoding' in response.headers:
            return response
    #print(f"{np.amax(data)}")
    gzip_buffer = io.BytesIO()
    gzip_file = gzip.GzipFile(mode='wb', compresslevel=5, fileobj=gzip_buffer)
    gzip_file.write(response.data)
    gzip_file.close()
    response.data = gzip_buffer.getvalue()

    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(response.data)

    return response

@app.route('/<path:dataset_name>/mesh/<int:id>:0')
def get_mesh_info(dataset_name,id):
    mesh_info= {"fragments": [f"{id}.ngmesh"]}
    return jsonify(mesh_info)

@app.route('/<path:dataset_name>/mesh/<int:id>.ngmesh')
def get_mesh(dataset_name,id):
    if("_n5ngBinarize" in dataset_name):
        dataset_name = dataset_name.replace("_n5ngBinarize","")

    print(f"getting mesh for {id}")
    return redirect(f"http://10.150.100.248:9000/{dataset_name}/mesh/{id}.ngmesh")
    #return temp

def main():
    global source_n5
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='n5 file to share', default='sample.n5')
    args = parser.parse_args()
    print(f"path: {args.filename}")
    source_n5 = args.filename

    # n5f = z5py.file.N5File(args.filename, mode='r')
    n5f = zarr.open(args.filename, mode='r')

    # Start flask
    app.debug = True
    app.config['n5file'] = n5f
    try:
         app.run(host='0.0.0.0')
    except:
         app.run(host='0.0.0.0',port=5001)

if __name__ == '__main__':
    main()