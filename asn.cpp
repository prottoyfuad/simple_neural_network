
#include <bits/stdc++.h>

int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout << std::fixed << std::setprecision(5);

  const int tests = 4;
  const int input_nodes = 2;
  const int output_nodes = 1;
  const int layers = 3;
  const int layer_nodes = input_nodes;
  
  const std::vector<std::vector<double>> inputs = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
  };
  const std::vector<std::vector<double>> outputs = {
    {0.0}, 
    {1.0},
    {1.0}, 
    {0.0}
  };
  
  auto sigmoid = [](double x) {
    return 1.0 / (1.0 + 1.0 / exp(x));
  };

  std::mt19937 rng (
    []() -> uint32_t {
      char* foo = new char;
      delete foo;
      return uint64_t(foo);
    }()
  );
  auto get_rand = [&rng] (double l, double r) {
    std::uniform_real_distribution<> dis(l, r);
    return dis(rng);
  };
  
  int n = input_nodes + output_nodes + (layers - 2) * layer_nodes;
  int m = layer_nodes * (input_nodes + output_nodes + (layers - 3) * layer_nodes);
  struct Edge {
    int id;
    int from, to;
    double weight;
    
    Edge(int i = -1) : id(i) {}
  };
  
  std::vector<std::vector<int>> Ein(n);
  std::vector<std::vector<int>> Eout(n);
  std::vector<Edge> edges;
  edges.reserve(m);
  
  for (int i = 0; i < input_nodes; i++) {
    for (int j = 0; j < layer_nodes; j++) {
      Edge edge(edges.size());
      edge.from = i;
      edge.to = j + input_nodes;
      edges.push_back(edge);
      
      Eout[edge.from].push_back(edge.id);
      Ein[edge.to].push_back(edge.id);
    }
  }  
  for (int k = 0; k < layers - 3; k++) {
    for (int i = 0; i < layer_nodes; i++) {
      for (int j = 0; j < layer_nodes; j++) {
        Edge edge(edges.size());
        edge.from = input_nodes + k * layer_nodes + i;
        edge.to = input_nodes + (k + 1) * layer_nodes + j;
        edges.push_back(edge);
        
        Eout[edge.from].push_back(edge.id);
        Ein[edge.to].push_back(edge.id);
      }
    }
  }  
  for (int i = 0; i < layer_nodes; i++) {
    for (int j = 0; j < output_nodes; j++) {
      Edge edge(edges.size());
      edge.from = input_nodes + (layers - 3) * layer_nodes + i;
      edge.to = input_nodes + (layers - 2) * layer_nodes + j;
      edges.push_back(edge);
      
      Eout[edge.from].push_back(edge.id);
      Ein[edge.to].push_back(edge.id);
    }
  }
  
  assert((int) edges.size() == m);
  for (int i = 0; i < m; i++) {
    edges[i].weight = get_rand(-1.9, 1.9);
  }  
  const double alpha = get_rand(0.1, 0.2);
  std::vector<double> threshold(n);
  for (int i = input_nodes; i < n; i++) {
    threshold[i] = get_rand(-0.1, 0.3);
  }
  /*
  //@ input from book(for testing) ::
      edges[0].weight = 0.5;
      edges[1].weight = 0.9;
      edges[2].weight = 0.4;
      edges[3].weight = 1.0;
      edges[4].weight = -1.2;
      edges[5].weight = 1.1;
  
      threshold[2] = 0.8;
      threshold[3] = -0.1;
      threshold[4] = 0.3;
      alpha = 0.1
  */
  
  auto run_test = [&](int tc) -> std::vector<double> {
    std::vector<double> res(n);
    for (int i = 0; i < n; i++) {
      if (i < input_nodes) res[i] = inputs[tc][i];
      else res[i] = sigmoid(res[i] - threshold[i]);
      for (int j : Eout[i]) {
        auto [id, from, to, weight] = edges[j];
        assert(from == i);
        res[to] += res[from] * weight;
      }
    }
    return res;
  };
  
  auto train = [&](const std::vector<double>& Y, std::vector<double>& error) {    
    std::vector<double> gradient(n);
    std::vector<double> weight_delta(m);
    std::vector<double> threshold_delta(n);
    for (int i = n - 1; i >= 0; i--) {
      double next_gradients = 0;
      for (int j : Eout[i]) {
        auto [id, from, to, weight] = edges[j];
        assert(from == i);
        weight_delta[id] = alpha * Y[from] * gradient[to];
        
        next_gradients += weight * gradient[to];
      }
      if (i + output_nodes >= n) {
        assert(next_gradients == 0);
        assert(output_nodes - n + i >= 0);
        next_gradients = error[output_nodes - n + i];
      }
      gradient[i] = Y[i] * (1.0 - Y[i]) * next_gradients;
      threshold_delta[i] = -1.0 * alpha * gradient[i];
    }
    for (int i = 0; i < n; i++) {
      threshold[i] += threshold_delta[i];
    }
    for (int i = 0; i < m; i++) {
      assert(edges[i].id == i);
      edges[i].weight += weight_delta[i];
    }
  };

  std::cout << "Before training errors : \n";
  double sum_err = 0;
  for (int tc = 0; tc < tests; tc++) {
    std::vector<double> Y = run_test(tc);
    std::vector<double> error(output_nodes);
    for (int i = 0; i < output_nodes; i++) {
      double actual = outputs[tc][i];
      double current = Y[n - output_nodes + i];
      error[i] = actual - current;
      sum_err += error[i] * error[i];
    }
    for (double e : error) std::cout << e << ' ';
    std::cout << '\n';
  }
  std::cout << "And sum of squared errors : " << sum_err << '\n';

  int iter = 0;
  const double limit = 1e-5;
  while (sum_err >= limit) {
    ++iter;
    sum_err = 0;
    for (int tc = 0; tc < tests; tc++) {
      std::vector<double> Y = run_test(tc);
      std::vector<double> error(output_nodes);
      for (int i = 0; i < output_nodes; i++) {
        double actual = outputs[tc][i];
        double current = Y[n - output_nodes + i];
        error[i] = actual - current;
        sum_err += error[i] * error[i];
      }
      train(Y, error);
    }
  }

  std::cout << "\nAfter " << iter << " iterations, errors : \n";
  sum_err = 0;
  for (int tc = 0; tc < tests; tc++) {
    std::vector<double> Y = run_test(tc);
    std::vector<double> error(output_nodes);
    for (int i = 0; i < output_nodes; i++) {
      double actual = outputs[tc][i];
      double current = Y[n - output_nodes + i];
      error[i] = actual - current;
      sum_err += error[i] * error[i];
    }
    for (double e : error) std::cout << e << ' ';
    std::cout << '\n';
  }
  std::cout << "And sum of squared errors : " << sum_err << '\n';
  return 0;
}
 
