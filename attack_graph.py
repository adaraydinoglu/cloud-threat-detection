import networkx as nx

def build_iam_attack_graph():
    G = nx.DiGraph()
    
    G.add_edge("User:intern_dev", "Role:DevRole", action="sts:AssumeRole")
    G.add_edge("Role:DevRole", "Role:SecOpsRole", action="sts:AssumeRole")
    G.add_edge("Role:SecOpsRole", "Action:DeleteTrail", action="cloudtrail:DeleteTrail")
    G.add_edge("Role:SecOpsRole", "Action:StopLogging", action="cloudtrail:StopLogging")
    
    G.add_edge("User:contractor", "Role:AnalyticsRole", action="sts:AssumeRole")
    G.add_edge("Role:AnalyticsRole", "Action:CreateTable", action="glue:CreateTable")
    G.add_edge("Role:AnalyticsRole", "Action:StartQueryExecution", action="athena:StartQueryExecution")
    
    return G

def find_attack_paths(graph, target_action):
    paths = []
    target_node = f"Action:{target_action}"
    
    if not graph.has_node(target_node):
        return paths
        
    for source in graph.nodes():
        if source.startswith("User:"):
            if nx.has_path(graph, source, target_node):
                all_paths_from_source = list(nx.all_simple_paths(graph, source, target_node))
                paths.extend(all_paths_from_source)
                
    return paths

if __name__ == "__main__":
    graph = build_iam_attack_graph()
    print(f"Graf Olusturuldu: {graph.number_of_nodes()} dugum, {graph.number_of_edges()} kenar.\n")
    
    test_actions = ["DeleteTrail", "CreateTable", "DescribeInstances"]
    
    for action in test_actions:
        paths = find_attack_paths(graph, action)
        print(f"Hedef Eylem: {action}")
        if paths:
            print(f"  Bulunan Saldiri Yolu Sayisi: {len(paths)}")
            for p in paths:
                print(f"  Rota: {' -> '.join(p)}")
        else:
            print("  Bu eyleme ulasan bilinen bir IAM saldiri yolu bulunamadi.")
        print("-" * 50)
