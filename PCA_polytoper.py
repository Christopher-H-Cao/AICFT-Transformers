import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from tensorboard.plugins import projector
import pandas as pd
import math
import sys

steinmanns={'a':'d', 'b':'e', 'c':'f', 'd':'a', 'e':'b','f':'c'}
def plot_convex_hull(points, labels):
    hull = ConvexHull(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    seen = []
    
    # Plot vertices
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])    
    ax.set_zticks([-1.0, -0.5, 0, 0.5, 1.0])    
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])    
    """
    for vec, label in zip(points[:,:3], labels):
        for vec2, label2 in zip(points[:,:3], labels):
            if label2 == label: continue
            #if steinmanns[label2[0]]==label[0]: continue
            vecs=np.arccos((np.dot(vec,vec2)/(np.linalg.norm(vec)*np.linalg.norm(vec2))))*(180/np.pi)
            print(label, label2, vecs)
            for vec3, label3 in zip(points[:,:3], labels):
                if label3 == label: continue
                #if steinmanns[label3[0]]==label[0]: continue
                if label3 == label2: continue
                #if steinmanns[label3[0]]==label2[0]: continue
                mytup = ''.join(sorted([label[0],label2[0],label3[0]]))
                if mytup in seen: continue
                else:
                    seen.append(mytup)
                    diffvec1 = np.array(vec2) - np.array(vec)
                    diffvec2 = np.array(vec3) - np.array(vec2)
                    diffvec3 = np.array(vec) - np.array(vec3)
                    angle1 = 180-np.arccos((np.dot(diffvec1,diffvec2)/(np.linalg.norm(diffvec1)*np.linalg.norm(diffvec2))))*(180/np.pi)
                    angle2 = 180-np.arccos((np.dot(diffvec2,diffvec3)/(np.linalg.norm(diffvec2)*np.linalg.norm(diffvec3))))*(180/np.pi)
                    angle3 = 180-np.arccos((np.dot(diffvec3,diffvec1)/(np.linalg.norm(diffvec3)*np.linalg.norm(diffvec1))))*(180/np.pi)
                    print(f"{label2}-{label} vs {label3}-{label2} vs {label}-{label3}")
                    print(angle1+angle2+angle3)
                    print(angle1)
                    print(angle2)
                    print(angle3)
    """
    # Plot faces
    for simplex in hull.simplices:
        s = hull.points[simplex]
        s = np.vstack((s, s[0]))
        ax.plot(s[:, 0], s[:, 1], s[:, 2],'k-')
    faces = Poly3DCollection([hull.points[simplex] for simplex in hull.simplices], alpha=0.5)
    faces.set_facecolor('xkcd:pale blue')
    ax.add_collection3d(faces)
    for i, label in enumerate(labels):
        ax.text(points[i, 0], points[i, 1], points[i, 2], label[0], fontsize='xx-large')
    #plt.axis('off')
    plt.show()

def angle_in_space(points, labels):
    seen=[]
    print(points.shape)
    for vec, label in zip(points, labels):
        for vec2, label2 in zip(points, labels):
            if label2 == label: continue
            vecs=np.arccos((np.dot(vec,vec2)/(np.linalg.norm(vec)*np.linalg.norm(vec2))))*(180/np.pi)
            if steinmanns[label2[0]]==label[0]:
                print(label, label2, vecs)
            for vec3, label3 in zip(points, labels):
                if label3 == label: continue
                #if steinmanns[label3[0]]==label[0]: continue
                if label3 == label2: continue
                #if steinmanns[label3[0]]==label2[0]: continue
                mytup = ''.join(sorted([label[0],label2[0],label3[0]]))
                if mytup in seen: continue
                else:
                    seen.append(mytup)
                    diffvec1 = np.array(vec2) - np.array(vec)
                    diffvec2 = np.array(vec3) - np.array(vec2)
                    diffvec3 = np.array(vec) - np.array(vec3)
                    angle1 = 180-np.arccos((np.dot(diffvec1,diffvec2)/(np.linalg.norm(diffvec1)*np.linalg.norm(diffvec2))))*(180/np.pi)
                    angle2 = 180-np.arccos((np.dot(diffvec2,diffvec3)/(np.linalg.norm(diffvec2)*np.linalg.norm(diffvec3))))*(180/np.pi)
                    angle3 = 180-np.arccos((np.dot(diffvec3,diffvec1)/(np.linalg.norm(diffvec3)*np.linalg.norm(diffvec1))))*(180/np.pi)
                    print(f"{label2}-{label} vs {label3}-{label2} vs {label}-{label3}")
                    print("mean:"+str((angle1+angle2+angle3)/3))
                    print("std:"+str(np.std([angle1,angle2,angle3])/np.sqrt(3)))
                    print(angle1)
                    print(angle2)
                    print(angle3)


if __name__ == '__main__':
    exp=sys.argv[1]
    tokens_tsv = f"../octahedra/{exp}_token_embs.tsv"
    metadata_tsv = f"../octahedra/{exp}_token_metadata.tsv"
    data = pd.read_csv(tokens_tsv, sep="\t", header=None).values   # read dummy .tsv file into memory
    metadata = pd.read_csv(metadata_tsv, sep="\t", header=None).values   # read dummy .tsv file into memory
    angle_in_space(data,metadata)
    pca = PCA(n_components=3)
    embedding = pca.fit_transform(data)
    print(pca.explained_variance_ratio_)
    plot_convex_hull(embedding,metadata)
