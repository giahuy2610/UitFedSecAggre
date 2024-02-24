# The customized Client class for Homomorphic encryption system

from Pyfhel import Pyfhel, PyCtxt
import numpy as np
from logging import WARNING
from flwr.common.logger import log
import datetime
from client import Client

class ClientHE(Client):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id, epochs) -> None:
        super().__init__(model, X_train, y_train, X_test, y_test, client_id)
        
        # 
        self.epochs = epochs
        
        # The model weight template is used to store the list of layer's shapes.
        model_weight_template = []
        for k in self.model.get_weights():
            model_weight_template.append(k.shape)
        self.model_weight_template = model_weight_template

    def load_pyfhel(self):
        HE = Pyfhel()
        HE.load_context("./context")
        HE.load_public_key("./pub.key")
        HE.load_secret_key("./sec.key")
        HE.load_rotate_key("./rotate.key")
        return HE

    def calculate_shape_of_tensor(self, size):
        # size = 173056
        # The max_y is constant number
        # 16384 = (2^15)/2, 15 is the maximum Polynomial modulus degree
        # ref:https://pyfhel.readthedocs.io/en/latest/_autoexamples/Demo_3_Float_CKKS.html#sphx-glr-autoexamples-demo-3-float-ckks-py
        max_y = 16384

        for y in range(1, max_y):
            if size % y == 0 and size // y < max_y:
                x = size // y
                break
        
        # y is the length of small array
        # x is the length of list of array
        # example: 1 array with length 173056 will be reshape to 13 array with length 13312
        return y, x

    def get_parameters(self, config):
        print("Malwarelient get_parameters"),datetime.datetime.now()

        HE = self.load_pyfhel()

        # Reshaping the tensors to 1 di `mensional numpy ndarray and dividing
        #  larger tensors relative to the HE context n attribute (see more comments below and see HE context gen)
        #  then encrypting each reshaped tensor and transforming them to bytes for sending them to the server.
        model_encrypted_weights_as_bytes = []
        for k in self.model.get_weights():
            try:     
                vector_length = np.prod(np.array(k.shape))
                reshaped_weight = k.reshape(vector_length,)
                
                # The Polynomial modulus degree (n in the HE context generation) is the limit
                #   for the length of the ndarray, so if we have larger arrays they have to be
                #   divided into smaller ones relative to the (2^15)/2
                if reshaped_weight.shape[0] > 16384:
                    # Reshape the tensor
                    x, y = self.calculate_shape_of_tensor(reshaped_weight.shape[0])
                    reshaped_weight = reshaped_weight.reshape((x,y))
                    
                    for large_weight in reshaped_weight:
                        # Each small array, encode to bytes
                        ptxt_x = HE.encodeFrac(large_weight.astype(np.float64))
                        encrypted_weight = HE.encryptPtxt(ptxt_x)
                        encrypted_weight_as_bytes = encrypted_weight.to_bytes()
                        
                        # Store to the array result
                        model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)
                        
                else:
                    # Encode to bytes
                    ptxt_x = HE.encodeFrac(reshaped_weight.astype(np.float64))
                    encrypted_weight = HE.encryptPtxt(ptxt_x)
                    encrypted_weight_as_bytes = encrypted_weight.to_bytes()
                    
                    # Store to the array result
                    model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)
                    
            except Exception as e:
                log(WARNING, str(e))
                log(WARNING, "nFAIL")
        
        return model_encrypted_weights_as_bytes

    def fit(self, parameters, config):
        print("Malwarelient fit",datetime.datetime.now())
        
        HE = self.load_pyfhel()

        # data preprocessing operations...
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
                x, y = self.calculate_shape_of_tensor(temp_shape_dproduct)
                
                for lto in range(0,x):
                    encr_bytes = parameters[offset_wit]
                    
                    # Decode the weights from byte to plaintext float
                    c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                    plaintext_weight = HE.decryptFrac(c_b)
                    plaintext_weight = plaintext_weight[:y]

                    large_tensor = np.concatenate((large_tensor, plaintext_weight))
                    offset_wit += 1

                large_tensor = large_tensor.reshape(model_weight_shape_obj)
                decrypted_weights.append(large_tensor)
            else:
                encr_bytes = parameters[offset_wit]
                c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                
                # Decode the weights from byte to plaintext float
                plaintext_weight = HE.decryptFrac(c_b)
                plaintext_weight = plaintext_weight[:temp_shape_dproduct]

                decrypted_weights.append(plaintext_weight.reshape(model_weight_shape_obj))
                offset_wit += 1

        self.model.set_weights(decrypted_weights)
        history = self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=32)

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
                x, y = self.calculate_shape_of_tensor(reshaped_weight.shape[0])
                reshaped_weight = reshaped_weight.reshape((x, y))
                
                for large_weight in reshaped_weight:
                    # Re-encode the new weight to bytes to send to server
                    ptxt_x = HE.encodeFrac(large_weight.astype(np.float64))
                    encrypted_weight = HE.encryptPtxt(ptxt_x)
                    encrypted_weight_as_bytes = encrypted_weight.to_bytes()

                    model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)

            else:
                # Re-encode the new weight to bytes to send to server
                ptxt_x = HE.encodeFrac(reshaped_weight.astype(np.float64))
                encrypted_weight = HE.encryptPtxt(ptxt_x)
                encrypted_weight_as_bytes = encrypted_weight.to_bytes()

                model_encrypted_weights_as_bytes.append(encrypted_weight_as_bytes)
                
        res_dw = {
            "client_id": self.client_id,
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "client_address": self.client_address,            
        }
        return model_encrypted_weights_as_bytes, len(self.X_train), res_dw
    
    def evaluate(self, parameters, config):
        print("MalwarelientHE evaluate",datetime.datetime.now())
        
        HE = self.load_pyfhel()

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
                
                x, y = self.calculate_shape_of_tensor(temp_shape_dproduct)
                for lto in range(0,x):
                    encr_bytes = parameters[offset_wit]
                    
                    c_b = PyCtxt(pyfhel=HE, bytestring=encr_bytes)
                    plaintext_weight = HE.decryptFrac(c_b)
                    plaintext_weight = plaintext_weight[:y]

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
    