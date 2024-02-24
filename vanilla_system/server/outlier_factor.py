import tensorflow as tf

def cosine_similarity(vector1, vector2):
    # Chuyển đổi vector thành các tensor trong TensorFlow
    v1 = tf.convert_to_tensor(vector1, dtype=tf.float32)
    v2 = tf.convert_to_tensor(vector2, dtype=tf.float32)

    # Reshape vector để có shape (batch_size, 1)
    v1 = tf.reshape(v1, (1, -1))
    v2 = tf.reshape(v2, (1, -1))

    # Tính cosine similarity
    similarity_loss = tf.keras.losses.CosineSimilarity(axis=1)(v1, v2)

    return abs(similarity_loss.numpy())


def cosine_similarity_normalization(client_distance_array):
    new_client_distance_array = []
    for client_distance in client_distance_array:
        new_client_distance_array.append((client_distance-min(client_distance_array))/(max(client_distance_array)-min(client_distance_array)))
    
    return new_client_distance_array


def outlier_factor():
    return
