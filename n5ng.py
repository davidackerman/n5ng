#!/usr/bin/env python3

import argparse
import gzip
import io
import zarr
import numpy as np
from flask import Flask, jsonify, Response, request, redirect
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
global source_n5

def get_scales(dataset_name, scales, encoding='raw', base_res=np.array([1.0,1.0,1.0])):
    def get_scale_for_dataset(dataset, scale, base_res):
        if 'resolution' in dataset.attrs:
            resolution = dataset.attrs['resolution']
        elif 'downsamplingFactors' in dataset.attrs:
            # The FAFB n5 stack reports downsampling, not absolute resolution
            resolution = (base_res * np.asarray(dataset.attrs['downsamplingFactors'])).tolist()
        else:
            resolution = (base_res*2**scale).tolist()
        print(f"{dataset.attrs}")
        return {
                    'chunk_sizes': [list(reversed(dataset.chunks))],
                    'resolution': resolution,
                    'size': list(reversed(dataset.shape)),
                    'key': str(scale),
                    'encoding': encoding,
                    'voxel_offset': dataset.attrs.get('offset', [0,0,0]),
                }

    if  scales:
        # Assumed scale pyramid with the convention dataset/sN, where N is the scale level
        scale_info = []
        for scale in scales:
            try:
                print("%s/s%d" % (dataset_name, scale))
                dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
                dataset = app.config['n5file'][dataset_name_with_scale]
                this_scale = scale_info.append(get_scale_for_dataset(dataset, scale, base_res))
            except Exception as exc:
                print(exc)
    else:
        dataset = app.config['n5file'][dataset_name]
        # No scale pyramid for this dataset
        scale_info = [ get_scale_for_dataset(dataset, 1.0, base_res) ]
    return scale_info

@app.route('/<path:dataset_name>/info')
def dataset_info(dataset_name):
    if "mesh" in dataset_name:
        info = {"@type": "neuroglancer_legacy_mesh"
                #"@type": "neuroglancer_multilod_draco",
                #"vertex_quantization_bits": 16,
                #"lod_scale_multiplier": 1
        }
    elif "_properties" in dataset_name:
        dataset = dataset_name.split("_properties")[0]
        print(dataset)
        meshes = os.listdir(source_n5+"/"+dataset+"/mesh")
        meshes = [mesh.split(".")[0] for mesh in meshes]
        info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
        "ids": meshes,
        "properties":[
            {"id": "label",
            "type": "label",
            "values": meshes
            },
            {
            "id": "description",
            "type": "description",
            "values": meshes
            }
            ]
        }
        }
    else:# elif "s0" in dataset_name or "s1" in dataset_name or "s2" in dataset_name:

      info = {
            'data_type' : 'uint8' if ( ("medialSurface" in dataset_name ) or ("sheet" in dataset_name) or ("binarized" in dataset_name)) else 'uint64',
            'type': 'segmentation',
            'num_channels' : 1,
            'mesh' : 'mesh',
            'scales' : get_scales(dataset_name, scales=list(range(0,3)), base_res=np.array([2.0, 2.0, 2.0])) if ("training" in dataset_name) else get_scales(dataset_name, scales=list(range(0,3)), base_res=np.array([4.0, 4.0, 4.0])) # get_scales(dataset_name, scales=list(range(0,8)), base_res=np.array([4.0, 4.0, 40.0]))
        }
    print(info)
    return jsonify(info)


# Implement the neuroglancer precomputed filename structure for URL requests
@app.route('/<path:dataset_name>/<int:scale>/<int:x1>-<int:x2>_<int:y1>-<int:y2>_<int:z1>-<int:z2>')
def get_data(dataset_name, scale, x1, x2, y1, y2, z1, z2):
    # TODO: Enforce a data size limit
    dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
    dataset = app.config['n5file'][dataset_name_with_scale]
    print(dataset)
    data = dataset[z1:z2,y1:y2,x1:x2]
    # Neuroglancer expects an x,y,z array in Fortram order (e.g., z,y,x in C =)
    response = Response(data.tobytes(order='C'), mimetype='application/octet-stream')

    accept_encoding = request.headers.get('Accept-Encoding', '')
    if 'gzip' not in accept_encoding.lower() or \
           'Content-Encoding' in response.headers:
            return response
    print(f"{np.amax(data)}")
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