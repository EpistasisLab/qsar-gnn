from .model import HeteroRGCNLayer, HeteroDotProductPredictor

class LinkPredictor(nn.Module):
    """Relational GCN for heterogeneous graphs designed to predict
    missing edges between chemicals and Tox21 assays.

    Parameters
    ----------
    in_size_dict : dict
        A dictionary mapping edge types to their input size. For any given edge
        type, the input size is equal to the dimensionality of the source
        node's feature matrix.
    hidden_size : int
        The dimensionality of the hidden (encoded) representation of nodes.
        Currently, this is a single, fixed dimensionality, but it may be
        extended to allow for different hidden sizes for each node type.
    out_size : int
        The dimensionality of nodes' output representations.
    rel_names : list of str
        List of all edge type names in the graph. These should be the same as
        the edge types described in `in_size_dict` (consider removing to lessen
        redundancy).
    """
    def __init__(self, in_size_dict, hidden_size, out_size, rel_names):
        super(LinkPredictor, self).__init__()

        # NOTE: May need to modify to take _one_ rel name
        self.sage = HeteroRGCNLayer(in_size_dict, out_size, rel_names)

        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        # An 'encoder', of sorts
        h = self.sage(g, x)

        return self.pred(g, h, etype), self.pred(neg_g, h, etype)


def compute_lp_loss(pos_score, neg_score):
    """Compute the Margin Loss between positive and negative edges for Link
    Prediction.
    
    Parameters
    ----------
    pos_score : torch.tensor
    neg_score : torch.tensor
    """
    # Margin loss:
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

