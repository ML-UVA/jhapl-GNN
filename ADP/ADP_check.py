import pickle
import networkx

def data_summary(data_path):
    adp_dict_path = os.path.join(data_path, "adp_dict.pkl")
    with open(adp_dict_path, "rb") as f:
        adp_dict = pickle.load(f)

    total_pairs = sum(len(dend_dict) for dend_dict in adp_dict.values())
    print(f"Total neuron pairs with ADP values: {total_pairs}" )

with open("../data/adp_data.pkl", "rb") as f:
    adp_dict = pickle.load(f)

with open("../data/ADP_graph_5_micron_threshold.pkl","rb") as f:
    ADP_network_x_graph = pickle.load(f)

print(list(ADP_network_x_graph.edges(data=True))[:10])

print(adp_dict['864691135334559465_0']['864691135888491657_0'])

if __name__ == "__main__":
    data_patha_path)