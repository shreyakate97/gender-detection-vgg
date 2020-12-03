import caffe

net = caffe.Net('VGG_FACE_deploy.prototxt', caffe.TEST)

# print out the network architecture to convert it to keras
print([(k, v.data.shape) for k, v in net.blobs.items()])


