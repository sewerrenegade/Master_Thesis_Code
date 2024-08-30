
def identity_function(x):
    return x
class EmbeddingFunction:
    from typing import Any
    import umap
    import phate
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap,TSNE
    DEFAULT_TRANSFROM_DICT = {
        "PHATE": (phate.PHATE,{
        'n_components': 8,
        'knn': 10,
        'decay': 40,
        't': 'auto'}),
        
        "TSNE": (TSNE,{
        'n_components': 8,
        'method':'exact'}),
        
        "Isomap": (Isomap,{
        'n_components': 8}),
        
        "UMAP": (umap.UMAP,{
        'n_components': 8}),
        
        "PCA": (PCA,{
        'n_components': 8})
    }
    def __init__(self,function_name= None,function_settings = None) -> None:
        self.name = function_name
        if self.name is None:
            self.name = "identity_function"
            self.settings = {}
            self.function = identity_function
        else:
            self.settings = function_settings
            if function_settings is None:
                self.settings = EmbeddingFunction.DEFAULT_TRANSFROM_DICT[self.name][1]
            
            self.function = EmbeddingFunction.DEFAULT_TRANSFROM_DICT[self.name][0](**self.settings).fit_transform
        
    def __call__(self,x) -> Any:
        return self.function(x)
    
    def to_dict(self):
        return {"function_name": self.name,"function_settings":self.settings}
    