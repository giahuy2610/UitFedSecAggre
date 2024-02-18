import flwr as fl
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import numpy as np
from logging import WARNING, INFO
from flwr.common.logger import log
import datetime

class MalwarelientHE(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id, epochs) -> None:
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_id = client_id
        self.model = model
        self.client_address = ''
        model_weight_template = []

        for k in self.model.get_weights():
            print(k.shape)
            model_weight_template.append(k.shape)
        self.model_weight_template = model_weight_template
        self.epochs = epochs


    def get_parameters(self, config):
        print("Malwarelient get_parameters"),datetime.datetime.now()
        print(config)
        log(INFO, "get params start")

        HE = Pyfhel() # Empty creation
        HE.load_context("./context")
        HE.load_public_key("./pub.key")
        HE.load_secret_key("./sec.key")
        HE.load_rotate_key("./rotate.key")

        # Reshaping the tensors to 1 dimensional numpy ndarray and dividing
        #  larger tensors relative to the HE context n attribute (see more comments below and see HE context gen)
        #  then encrypting each reshaped tensor and transforming them to bytes for sending them to the server.
        model_encrypted_weights_as_bytes = []
        for k in self.model.get_weights():
            print("each layer in model------------------------------------------------")
            try:     
                vector_length = np.prod(np.array(k.shape))
                reshaped_weight = k.reshape(vector_length,)
                print("k shape", k.shape)
                print("vector length: ", vector_length)
                
                # The Polynomial modulus degree (n in the HE context generation) is the limit
                #   for the length of the ndarray, so if we have larger arrays they have to be
                #   divided into smaller ones relative to the (2^15)/2
                if reshaped_weight.shape[0] > 16384:
                    print("reshape > 16384")
                    reshaped_weight = reshaped_weight.reshape((13, 13312))
                    for large_weight in reshaped_weight:
                        print("divided each reshaped to list of array")
                        ptxt_x = HE.encodeFrac(large_weight.astype(np.float64))
                        encrypted_weight = HE.encryptPtxt(ptxt_x)
                        print("sos test", type(encrypted_weight))
                        encrypted_weight_as_bytes = encrypted_weight.to_bytes()
                        model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)
                        print("model_encrypted_weights_as_bytes current length", len(model_encrypted_weights_as_bytes))
                        
                else:
                    print("reshape <= 16384")
                    ptxt_x = HE.encodeFrac(reshaped_weight.astype(np.float64))
                    encrypted_weight = HE.encryptPtxt(ptxt_x)
                    encrypted_weight_as_bytes = encrypted_weight.to_bytes()
                    model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)
                    print("model_encrypted_weights_as_bytes current length", len(model_encrypted_weights_as_bytes))
                    
            except Exception as e:
                log(WARNING, str(e))
                log(WARNING, "nFAIL")
        
        
        a = np.ndarray((2,), buffer=np.array([1,2,3]),
           offset=np.int_().itemsize,
           dtype=int)
        c_b = PyCtxt(pyfhel=HE, bytestring=model_encrypted_weights_as_bytes[0])
        plaintext_weight = HE.decryptFrac(c_b)
        print("sos 11", type(model_encrypted_weights_as_bytes[0]))
        print(plaintext_weight)
        print(type(a))
        print(type(plaintext_weight))
        print(len(model_encrypted_weights_as_bytes))
        return model_encrypted_weights_as_bytes


    def fit(self, parameters, config):
        print("Malwarelient fit",datetime.datetime.now())
        print("FLclient paramters", len(parameters))
        HE = Pyfhel() # Empty creation
        HE.load_context("./context")
        HE.load_public_key("./pub.key")
        HE.load_secret_key("./sec.key")
        HE.load_rotate_key("./rotate.key")

        print("sos fit paramter", type(parameters[0]))
        
        # data preprocessing operations...
        decrypted_weights = []
        offset_wit = 0

        # Reading the received bytes and transforming them to HE cyphertexts
        #   which then are decrypted and reshaped to the original model shape
        for model_weight_shape_obj in self.model_weight_template:
            print("decrypt layer", offset_wit, model_weight_shape_obj)
            temp_shape_dproduct = np.prod(np.array(model_weight_shape_obj))

            # The Polynomial modulus degree (n in the HE context generation) is the limit
            #   for the length of the ndarray, so if we have larger arrays they have to be
            #   divided into smaller ones relative to the (2^15)/2
            #   The large tensor was divided into 8 (the iterator length) ndarrays with dimension 15744. 
            #   This should be defined dynamically, but is hardcoded for now
            if temp_shape_dproduct > 16384:
                print("decrypt case 1")
                large_tensor = np.array([])
                for lto in range(0,13):
                    print("decrypt the element at index", offset_wit)
                    encr_bytes = parameters[offset_wit]
                    c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                    plaintext_weight = HE.decryptFrac(c_b)

                    # TODO: think about dynamic calculation here
                    plaintext_weight = plaintext_weight[:13312]

                    large_tensor = np.concatenate((large_tensor, plaintext_weight))
                    offset_wit += 1

                large_tensor = large_tensor.reshape(model_weight_shape_obj)
                decrypted_weights.append(large_tensor)
            else:
                print("sos case 2")
                encr_bytes = parameters[offset_wit]
                c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                plaintext_weight = HE.decryptFrac(c_b)

                plaintext_weight = plaintext_weight[:temp_shape_dproduct]

                decrypted_weights.append(plaintext_weight.reshape(model_weight_shape_obj))
                offset_wit += 1

        self.model.set_weights(decrypted_weights)
        history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=32, validation_split=0.2,)

        # Reshaping the tensors to 1 dimensional numpy ndarray and dividing
        #  larger tensors relative to the HE context n attribute (see more comments below and see HE context gen)
        #  then encrypting each reshaped tensor and transforming them to bytes for sending them to the server.
        model_encrypted_weights_as_bytes = []
        for k in self.model.get_weights():
            vector_length = np.prod(np.array(k.shape))

            reshaped_weight = k.reshape(vector_length,)
            
            # The Polynomial modulus degree (n in the HE context generation) is the limit
            #   for the length of the ndarray, so if we have larger arrays they have to be
            #   divided into smaller ones relative to the (2^15)/2
            #   The large tensor was divided into 8 (the iterator length) ndarrays with dimension 15744. 
            #   This should be defined dynamically, but is hardcoded for now
            if reshaped_weight.shape[0] > 16384:
                reshaped_weight = reshaped_weight.reshape((13, 13312))
                
                for large_weight in reshaped_weight:
                    ptxt_x = HE.encodeFrac(large_weight.astype(np.float64))
                    encrypted_weight = HE.encryptPtxt(ptxt_x)

                    encrypted_weight_as_bytes = encrypted_weight.to_bytes()

                    model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)

            else:
                ptxt_x = HE.encodeFrac(reshaped_weight.astype(np.float64))
                encrypted_weight = HE.encryptPtxt(ptxt_x)

                encrypted_weight_as_bytes = encrypted_weight.to_bytes()

                model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)
        c_b = PyCtxt(pyfhel=HE, bytestring=model_encrypted_weights_as_bytes[0])
        print("lol")
        print(len(model_encrypted_weights_as_bytes))
        print(len(self.X_train))
        res_dw = {
            "client_id": self.client_id,
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "client_address": self.client_address,            
        }
        return model_encrypted_weights_as_bytes, len(self.X_train), res_dw
    

    def evaluate(self, parameters, config):
        print("MalwarelientHE evaluate",datetime.datetime.now())
        HE = Pyfhel() # Empty creation
        HE.load_context("./context")
        HE.load_public_key("./pub.key")
        HE.load_secret_key("./sec.key")
        HE.load_rotate_key("./rotate.key")

        decrypted_weights = []
        offset_wit = 0

        # Reading the received bytes and transforming them to HE cyphertexts
        #   which then are decrypted and reshaped to the original model shape
        for model_weight_shape_obj in self.model_weight_template:

            temp_shape_dproduct = np.prod(np.array(model_weight_shape_obj))

            # The Polynomial modulus degree (n in the HE context generation) is the limit
            #   for the length of the ndarray, so if we have larger arrays they have to be
            #   divided into smaller ones relative to the (2^15)/2
            #   The large tensor was divided into 8 (the iterator length) ndarrays with dimension 15744. 
            #   This should be defined dynamically, but is hardcoded for now
            if temp_shape_dproduct > 16384:
                large_tensor = np.array([])
                for lto in range(0,13):
                    encr_bytes = parameters[offset_wit]
                    c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                    plaintext_weight = HE.decryptFrac(c_b)

                    # TODO: think about dynamic calculation here
                    plaintext_weight = plaintext_weight[:13312]

                    large_tensor = np.concatenate((large_tensor, plaintext_weight))
                    offset_wit += 1

                large_tensor = large_tensor.reshape(model_weight_shape_obj)
                decrypted_weights.append(large_tensor)
            else:
                encr_bytes = parameters[offset_wit]
                c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                plaintext_weight = HE.decryptFrac(c_b)

                plaintext_weight = plaintext_weight[:temp_shape_dproduct]

                decrypted_weights.append(plaintext_weight.reshape(model_weight_shape_obj))
                offset_wit += 1
        
        self.model.set_weights(decrypted_weights)

        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)

        return loss, len(self.X_test), {"accuracy": float(accuracy)}
    