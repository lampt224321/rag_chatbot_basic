import torch

class Config:
    # Model Configurations
    EMBEDDING_MODEL_ID = "bkai-foundation-models/vietnamese-bi-encoder" 

    """ Mô hình embedding này sẽ giúp chuyển các văn bản thành các
        vector. Các vector này sẽ được sử dụng trong quá trình semantic
        chunking để chia văn bản thành các phần nhỏ có ý nghĩa hơn,
        thay vì tách theo độ dài cố định. """

    LLM_MODEL_ID = "lmsys/vicuna-7b-v1.5" 
    
    # Semantic Chunking Parameters
    CHUNK_BUFFER_SIZE = 1
    CHUNK_BREAKPOINT_TYPE = "percentile"
    CHUNK_BREAKPOINT_AMOUNT = 95
    MIN_CHUNK_SIZE = 500
    
    # Generation Config 
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.2 
    
    # System Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32 