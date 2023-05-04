import img_proccessing
from sklearn.neighbors import NearestNeighbors

def poces_similar_sift_descriprors_ann_index_TEST():

    imgs = ['./photos/photo_1@04-03-2022_01-26-02.jpg',
            './photos/photo_2@04-03-2022_01-27-23.jpg',
            './photos/photo_3@04-03-2022_01-27-23.jpg',
            './photos/photo_4@04-03-2022_01-30-18.jpg',
            './photos/photo_5@04-03-2022_01-30-32.jpg',
            './photos/photo_6@04-03-2022_01-34-08.jpg',
            './photos/photo_2441@16-01-2023_16-02-28.jpg',
            './photos/photo_2442@16-01-2023_18-41-43.jpg',
            './photos/photo_2443@16-01-2023_18-41-49.jpg',
            './photos/photo_2444@16-01-2023_21-17-04.jpg',
            './photos/photo_2445@16-01-2023_23-48-08.jpg',
            './photos/photo_2446@17-01-2023_00-26-44.jpg',
            './photos/photo_2447@17-01-2023_00-30-41.jpg',
            './photos/photo_2448@17-01-2023_00-31-34.jpg',
            './photos/photo_2449@17-01-2023_01-30-26.jpg',
            './photos/photo_244@20-03-2022_16-20-13.jpg',
            './photos/photo_2450@17-01-2023_01-31-59.jpg',
            './photos/photo_2451@17-01-2023_01-33-07.jpg',
            './photos/photo_2452@18-01-2023_00-37-27.jpg',
            './photos/photo_2453@18-01-2023_01-25-20.jpg',
            './photos/photo_2454@18-01-2023_09-42-47.jpg',
            './photos/photo_2455@18-01-2023_09-49-05.jpg',
            './photos/photo_2456@18-01-2023_17-50-31.jpg',
    ]

    descriptors = []

    for path in imgs:
        img_data = img_proccessing.get_image_data(path)
        descriptors.append(img_data.descriptor)


    nfeatures = 5000
    res_descriptors = []


    n_neighbors = 5
    index = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree')

    for i, desc in enumerate(descriptors):
        index.fit(desc)

    query_descriptor = descriptors[5]
    print(imgs[5])
    distances, indices = index.kneighbors(query_descriptor[:500], n_neighbors=n_neighbors)

    print(indices, len(descriptors))
    for idx in indices.flatten():
        if idx < len(descriptors):
            res_descriptors.append((descriptors[idx], idx))

    print(len(res_descriptors))
    for des in res_descriptors:
        if img_proccessing.compare_sift_descriprtors(query_descriptor, des[0]) == True:
            print("got req desc", imgs[des[1]])



def main():

    # bild_index_TEST()
    # db_sphere_index_search_TEST()
    poces_similar_sift_descriprors_ann_index_TEST()

    # retrive_top_k_descriptors_TEST()
    # poces_similar_sift_descriprors_brootforce_TEST()
    # poces_similar_sift_descriprors_top_k_TEST()
    # compare_evc_dist_TEST()


if __name__ == "__main__":
    main()
