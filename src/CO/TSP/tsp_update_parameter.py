import os
import shutil
import numpy as np
import json
import random
import time
import pickle
import sys
import types
import re
import warnings
import http.client
import ast
import concurrent.futures
import heapq
from typing import Sequence, Tuple, Dict, Any, Optional, List
from joblib import Parallel, delayed
import inspect
import hashlib
import traceback
import statistics

# --- External/Local Dependencies ---

# Import external dependencies for TSP evaluation (as specified in the original script)
try:
    from gls.gls_run import solve_instance
except ImportError:
    print("Warning: 'gls.gls_run' not found. TSP evaluation will fail.")
    # Provide a fallback implementation if the external solver is missing
    def solve_instance(* args, **kwargs):
        print("Error: gls_run.solve_instance is missing. Returning penalty.")
        return float('inf')

# Import necessary libraries for parameter optimization
try:
    # Try importing requests first for robust HTTP handling
    import requests
except ImportError:
    # If requests is missing, rely on http.client (built-in)
    print("Info: 'requests' library not found. Using built-in http.client for LLM communication.")
    requests = None

try:
    import optuna
    from optuna.storages import RDBStorage, InMemoryStorage
except ImportError:
    print("Warning: Optuna not installed. Parameter optimization features will be disabled.")
    optuna = None

# Handle AST manipulation dependencies (required for code modification)
try:
    import astor
except ImportError:
    if sys.version_info < (3, 9):
        print("Warning: 'astor' library not found. AST manipulation requires astor for Python < 3.9.")
    astor = None

# Define a reliable way to convert AST back to source code
def ast_to_source(tree: ast.AST) -> str:
    try:
        if hasattr(ast, 'unparse'):
            return ast.unparse(tree).strip()
        elif astor:
            return astor.to_source(tree).strip()
    except Exception as e:
        print(f"Warning: AST to source conversion failed: {e}")
    return ""


# Assume selection and management are local libraries. (Requirement R3)
try:
    from selection import prob_rank,equal,roulette_wheel,tournament
    from management import pop_greedy,ls_greedy,ls_sa
except ImportError:
    # Provide placeholders if libraries are missing
    print("Warning: Local libraries 'selection' or 'management' not found. Execution will fail.")
    def placeholder_func(*args, **kwargs):
        raise NotImplementedError(f"Local library function called but implementation is missing.")
    
    prob_rank = equal = roulette_wheel = tournament = placeholder_func
    pop_greedy = ls_greedy = ls_sa = placeholder_func


# --- Configuration Class ---

class Paras():
    def __init__(self):
        #####################
        ### General settings  ###
        #####################
        self.method = 'eoh'                # Selected method
        self.problem = 'tsp_construct'     # Selected problem (TSP)
        self.selection = None              # Individual selection method
        self.management = None             # Population management method

        #####################
        ###  EC settings  ###
        #####################
        self.ec_pop_size = 5  # number of algorithms in each population
        self.ec_n_pop = 5 # number of populations
        self.ec_operators = None # evolution operators
        self.ec_m = 2  # number of parents for crossover operators
        self.ec_operator_weights = None
        
        #####################
        ### LLM settings  ###
        ##################### 
        self.llm_api_endpoint = None # endpoint for remote LLM
        self.llm_api_key = None  # API key for remote LLM
        self.llm_model = None  # model type for remote LLM

        #####################
        ###  Exp settings  ###
        #####################
        self.exp_debug_mode = False  # if debug

        base_output_path = os.getenv('TSP_OUTPUT_PATH', './tsp_output')
        self.exp_output_path = f"{base_output_path}"
        self.exp_n_proc = 1
        
        #####################
        ###  Evaluation settings  ###
        #####################
        self.eva_timeout = 600
        self.eva_numba_decorator = False # Numba usage (False by default for TSP)

        #####################
        ###  Parameter Optimization settings (Integrated) ###
        #####################
        self.opt_n_trials = 30 # Number of trials for Optuna optimization

        self.opt_range_factor = 0.1 # Factor for additive range around zero (e.g., [-0.1, 0.1])
        self.opt_storage_path = "sqlite:///eoh_tsp_optimization.db" # Storage for Optuna studies

    def set_parallel(self):
        try:
            import multiprocessing
            num_processes = multiprocessing.cpu_count()
            if self.exp_n_proc == -1 or (self.exp_n_proc > num_processes and num_processes > 0):
                self.exp_n_proc = num_processes
                print(f"Set the number of proc to {num_processes} .")
        except Exception:
            self.exp_n_proc = 1
            print("Multiprocessing setup failed. Setting number of proc to 1.")
    
    def set_ec(self):    
        # Logic from original TSP script
        if self.management == None:
            if self.method in ['ael','eoh']:
                self.management = 'pop_greedy'
            elif self.method == 'ls':
                self.management = 'ls_greedy'
            elif self.method == 'sa':
                self.management = 'ls_sa'
        
        if self.selection == None:
            self.selection = 'prob_rank'
            
        
        if self.ec_operators == None:
            if self.method == 'eoh':
                self.ec_operators  = ['e1','e2','m1','m2']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1, 1, 1]
            elif self.method == 'ael':
                self.ec_operators  = ['crossover','mutation']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1, 1]
            elif self.method in ['ls', 'sa']:
                self.ec_operators  = ['m1']
                if self.ec_operator_weights == None:
                    self.ec_operator_weights = [1]
                    
        if self.method in ['ls','sa'] and self.ec_pop_size >1:
            self.ec_pop_size = 1
            self.exp_n_proc = 1
            print("> single-point-based, set pop size to 1. ")
            
    def set_evaluation(self):
        # Settings specific to the problem
        if self.problem == 'tsp_construct':
            self.eva_timeout = 600
            # self.eva_numba_decorator remains as set in init or via set_paras
                
    def set_paras(self, *args, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
              
        # Initialize settings
        self.set_parallel()
        self.set_ec()
        self.set_evaluation()

# --- Utility Functions ---

def create_folders(results_path: str):
    folder_path = os.path.join(results_path, "results")

    # Create the main folder "results"
    os.makedirs(folder_path, exist_ok=True)

    # Create subfolders inside "results"
    subfolders = ["history", "pops", "pops_best"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

# --- AST Utility Functions ---

def add_import_package_statement(program: str, package_name: str, as_name: Optional[str]=None, *, check_imported: bool=True) -> str:
    """Add 'import package_name as as_name' in the program code using AST."""
    if not ast_to_source: return program

    try:
        tree = ast.parse(program)
    except SyntaxError:
        return program # Return original if parsing fails

    if check_imported:
        package_imported = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == package_name for alias in node.names):
                    package_imported = True
                    break
            elif isinstance(node, ast.ImportFrom):
                 if node.module == package_name:
                     package_imported = True
                     break

        if package_imported:
            return program

    # Create a new import node
    import_node = ast.Import(names=[ast.alias(name=package_name, asname=as_name)])
        
    # Insert the new import node near the top
    tree.body.insert(0, import_node)
    ast.fix_missing_locations(tree)
    program = ast_to_source(tree)
    return program


def _add_numba_decorator(
        program: str,
        function_name: str
) -> str:
    if not ast_to_source: return program

    # Ensure 'import numba' is present first
    program = add_import_package_statement(program, 'numba', check_imported=True)

    try:
        tree = ast.parse(program)
    except SyntaxError:
        return program
    
    for node in ast.walk(tree):
        # Find the target function definition
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Check if decorator already exists
            already_decorated = False
            for decorator in node.decorator_list:
                # Handle @numba.jit or @numba.jit(...)
                func = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(func, ast.Attribute) and func.attr == 'jit':
                         if isinstance(func.value, ast.Name) and func.value.id == 'numba':
                            already_decorated = True
                            break
            
            if not already_decorated:
                # Create the @numba.jit(nopython=True) decorator node
                decorator = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='numba', ctx=ast.Load()),
                        attr='jit',
                        ctx=ast.Load()
                    ),
                    args=[],
                    # Use ast.Constant for True
                    keywords=[ast.keyword(arg='nopython', value=ast.Constant(value=True))]
                )
                # Add the decorator to the function
                node.decorator_list.insert(0, decorator)

    ast.fix_missing_locations(tree)
    modified_program = ast_to_source(tree)
    return modified_program


def add_numba_decorator(
        program: str,
        function_name: Optional[str | Sequence[str]],
) -> str:
    if not function_name:
        return program
        
    if isinstance(function_name, str):
        return _add_numba_decorator(program, function_name)
    # Handle multiple function names
    for f_name in function_name:
        program = _add_numba_decorator(program, f_name)
    return program


# --- Problem Definition (TSP) ---

class GetPrompts():
    """Prompt definitions for TSP Guided Local Search Heuristic."""
    def __init__(self):
        # Task description from original TSP script
        self.prompt_task = "Task: Given an edge distance matrix and a local optimal route, please help me design a strategy to update the distance matrix to avoid being trapped in the local optimum with the final goal of finding a tour with minimized distance. \
You should create a heuristic for me to update the edge distance matrix."
        self.prompt_func_name = "update_edge_distance"
        self.prompt_func_inputs = ['edge_distance', 'local_opt_tour', 'edge_n_used']
        self.prompt_func_outputs = ['updated_edge_distance']
        self.prompt_inout_inf = "'local_opt_tour' includes the local optimal tour of IDs, 'edge_distance' and 'edge_n_used' are n * n matrices, 'edge_n_used' includes the number of each edge used during permutation."
        self.prompt_other_inf = "All are Numpy arrays."

    def get_task(self):
        return self.prompt_task
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self) -> List[str]:
        return self.prompt_func_inputs
    
    def get_func_outputs(self) -> List[str]:
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.prompt_inout_inf

    def get_other_inf(self):
        return self.prompt_other_inf

# --- Data Loading Utilities (TSP) ---
# (Requirement R2: Do not modify external data handling)

def read_coordinates(instance_path: str, file_name: str):
    coordinates = []
    optimal_distance = 1E10

    try:
        # Use os.path.join for robust path manipulation
        with open(os.path.join(instance_path, file_name), 'r') as file:
            lines = file.readlines()
            
        index = -1
        for i, line in enumerate(lines):
            if line.startswith('NODE_COORD_SECTION'):
                index = i + 1
                break
        
        if index == -1:
            return None, None

        for i in range(index, len(lines)):
            line_strip = lines[i].strip()
            if line_strip == 'EOF' or line_strip == '': 
                break
            parts = lines[i].split()
            # Expecting ID X Y
            if len(parts) >= 3:
                try:
                    coordinates.append([float(parts[1]), float(parts[2])])
                except ValueError:
                    continue # Skip malformed lines

        # Try reading solutions file
        sol_path = os.path.join(instance_path, "solutions")
        if os.path.exists(sol_path):
            with open(sol_path, 'r') as sol:
                sol_lines = sol.readlines()
            base_name = file_name.removesuffix(".tsp")
            for line in sol_lines:
                if line.startswith(base_name):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            optimal_distance = float(parts[2])
                        except ValueError:
                            pass
                    break

    except FileNotFoundError:
        print(f"Warning: Instance file {file_name} or solutions file not found in {instance_path}.")
        return None, None
    except Exception as e:
        print(f"Error reading instance {file_name}: {e}")
        return None, None

    return np.array(coordinates), optimal_distance

def create_distance_matrix(coordinates):
    # Calculates the Euclidean distance matrix
    if coordinates is None or len(coordinates) == 0:
        return np.array([])
    distance_matrix = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
    return distance_matrix

def read_instance(instance_path: str, filename: str):
    # Reads a single TSP instance
    coord, opt_cost = read_coordinates(instance_path, filename)
    if coord is None:
        return None, None, None
    instance = create_distance_matrix(coord)
    return coord, instance, opt_cost

def read_instance_all(instances_path: str):
    # Reads all TSP instances in the directory
    if not os.path.exists(instances_path):
        print(f"Warning: Training data path not found at {instances_path}")
        return [], [], [], []
        
    file_names = os.listdir(instances_path)
    coords = []
    instances = []
    opt_costs = []
    names = []
    for filename in file_names:
        if filename.endswith('.tsp'):
            coord, instance, opt_cost = read_instance(instances_path, filename)
            if coord is not None:
                coords.append(coord)
                instances.append(instance)
                opt_costs.append(opt_cost)
                names.append(filename)
    return coords, instances, opt_costs, names
    
class TSPGLS():
    """TSP Problem Definition and Evaluation Environment."""
    def __init__(self, debug_mode: bool=False) -> None:
        # Configuration from original TSP script
        self.n_inst_eva = 3 # A small value for test only
        self.time_limit = 10 # maximum 10 seconds for each instance
        self.ite_max = 1000 # maximum number of local searchs in GLS for each instance
        self.perturbation_moves = 1 
        
        # Path to training data (Requirement R2)
        self.instance_path = os.getenv('TSP_INSTANCE_PATH', './TrainingData') 
        self.debug_mode = debug_mode

        # Load instances
        self.coords,self.instances,self.opt_costs,self.names = read_instance_all(self.instance_path)
        
        if not self.instances:
             print(f"Warning: No TSP instances loaded from {self.instance_path}.")

        self.prompts = GetPrompts()

    # (Integrated) Wrapper class to handle parameterized heuristics transparently for solve_instance (R1/R5)
    class HeuristicWrapper:
        """Wraps the generated heuristic module to handle optional parameters."""
        def __init__(self, module, params: Optional[Dict], debug_mode: bool=False):
            self.module = module
            self.params = params if params is not None else {}
            self.func_name = 'update_edge_distance' # The target function name for TSP GLS
            self.target_func = getattr(self.module, self.func_name, None)
            self.debug_mode = debug_mode
            
            self.accepts_params = False
            if self.target_func:
                try:
                    # Use inspect to check the signature dynamically
                    sig = inspect.signature(self.target_func)
                    self.accepts_params = 'params' in sig.parameters
                except Exception:
                    pass

        # Mimic the module structure expected by solve_instance
        def update_edge_distance(self, edge_distance, local_opt_tour, edge_n_used):
            if not self.target_func:
                return edge_distance
            
            try:
                # Input validation to prevent common errors
                if edge_distance is None or local_opt_tour is None or edge_n_used is None:
                    if self.debug_mode:
                        print("Runtime error during execution of heuristic function: One or more inputs are None")
                    return edge_distance
                
                # Ensure all arrays are writable by making copies
                edge_distance_writable = np.array(edge_distance, copy=True)
                local_opt_tour_writable = np.array(local_opt_tour, copy=True)
                edge_n_used_writable = np.array(edge_n_used, copy=True)
                
                # Validate array dimensions
                if edge_distance_writable.size == 0 or local_opt_tour_writable.size == 0 or edge_n_used_writable.size == 0:
                    if self.debug_mode:
                        print("Runtime error during execution of heuristic function: One or more arrays are empty")
                    return edge_distance
                
                if np.any(np.array(edge_distance_writable.shape) < 0) or np.any(np.array(edge_n_used_writable.shape) < 0):
                    if self.debug_mode:
                        print("Runtime error during execution of heuristic function: Negative dimensions are not allowed")
                    return edge_distance
                
                if self.accepts_params:
                    # Call parameterized version (during optimization)
                    result = self.target_func(edge_distance_writable, local_opt_tour_writable, edge_n_used_writable, self.params)
                else:
                    # Call standard version (during evolution or after optimization)
                    result = self.target_func(edge_distance_writable, local_opt_tour_writable, edge_n_used_writable)
                
                # Validate the result
                if result is None:
                    if self.debug_mode:
                        print("Runtime error during execution of heuristic function: Function returned None")
                    return edge_distance
                
                result_array = np.array(result)
                if result_array.size == 0 or np.any(np.array(result_array.shape) < 0):
                    if self.debug_mode:
                        print("Runtime error during execution of heuristic function: Invalid result dimensions")
                    return edge_distance
                
                return result
                
            except ValueError as e:
                # Handle specific ValueError cases (e.g., "Cannot take a larger sample than population")
                if self.debug_mode:
                    print(f"Runtime error during execution of heuristic function: {e}")
                return edge_distance
            except Exception as e:
                # Handle errors during heuristic execution robustly
                if self.debug_mode:
                    print(f"Runtime error during execution of heuristic function: {e}")
                # Fallback strategy: return original distance matrix if heuristic fails
                return edge_distance

    # ADAPTED: Now accepts optional 'params'.
    def evaluateGLS(self, heuristic_module, params: Optional[Dict]=None) -> Optional[float]:
        
        if not self.instances:
            return None

        # Wrap the module so solve_instance can call it correctly regardless of parameterization
        heuristic_wrapper = self.HeuristicWrapper(heuristic_module, params, self.debug_mode)

        # Determine the number of instances to evaluate
        n_eval = min(self.n_inst_eva, len(self.instances))
        
        if n_eval == 0:
            return None

        gaps = np.zeros(n_eval)

        # Sequential evaluation
        for i in range(n_eval):
            try:
                # Call the external GLS solver
                gap = solve_instance(i,self.opt_costs[i],  
                                     self.instances[i], 
                                     self.coords[i],
                                     self.time_limit,
                                     self.ite_max,
                                     self.perturbation_moves,
                                     heuristic_wrapper) # Pass the wrapper
                gaps[i] = gap
            except Exception as e:
                # Handle errors during solve_instance
                if self.debug_mode:
                    print(f"Error evaluating instance {i} with GLS: {e}")
                # Assign a large penalty gap
                gaps[i] = float('inf')

        # Calculate mean gap
        # Filter out infinite gaps if some succeeded
        valid_gaps = gaps[np.isfinite(gaps)]
        if len(valid_gaps) == 0:
            return None # Indicate complete failure
            
        return np.mean(valid_gaps)
    

    # ADAPTED: Now accepts optional 'params' and 'n_evals' for robust evaluation.
    def evaluate(self, code_string: str, params: Optional[Dict]=None, n_evals: int=1) -> Optional[float]:
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Ensure numpy is available in the module's namespace (Crucial for TSP heuristics)
                heuristic_module.__dict__['np'] = np
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules (if needed by external libraries)
                sys.modules[heuristic_module.__name__] = heuristic_module

                # Perform evaluations
                fitness_values = []
                for eval_round in range(n_evals):
                    # Pass 'params' down to evaluateGLS
                    fitness = self.evaluateGLS(heuristic_module, params)
                    
                    if fitness is not None:
                        fitness_values.append(fitness)

                if not fitness_values:
                    return None # Indicates evaluation failure

                # Use the median fitness for robustness
                median_fitness = statistics.median(fitness_values)

                return median_fitness
            
        except Exception as e:
            # Error during execution of the generated code (e.g., syntax error)
            if self.debug_mode:
                print(f"DEBUG: Exception in evaluate wrapper (e.g., syntax/runtime error in generated code): {e}")
                # traceback.print_exc()
            return None


class Probs():
    """Loads or creates problem instances."""
    def __init__(self,paras: Paras):
        if not isinstance(paras.problem, str):
            # Load existing problem instance
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            # Create new TSP problem instance
            self.prob = TSPGLS(debug_mode=paras.exp_debug_mode)
            print("- Prob "+paras.problem+" loaded ")
        else:
            print("problem "+paras.problem+" not found!")
            self.prob = None

    def get_problem(self):
        return self.prob


# --- LLM Communication Module ---

class InterfaceAPI:
    """Handles API communication with LLM."""
    def __init__(self, api_endpoint: str, api_key: str, model_LLM: str, debug_mode: bool):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content: str) -> Optional[str]:
        # Use 'requests' if available, otherwise fall back to 'http.client'.
        if requests:
            return self._get_response_requests(prompt_content)
        else:
            return self._get_response_httpclient(prompt_content)

    def _get_response_requests(self, prompt_content: str) -> Optional[str]:
        payload = {
            "model": self.model_LLM,  
            "messages": [
                {"role": "user", "content": prompt_content}
            ],
            "temperature": 0.7 # Added temperature for diversity
        }
        
        headers = {
            "Authorization": "Bearer " + self.api_key,           
            "User-Agent": "EoH-AutoParams-Agent",   
            "Content-Type": "application/json",                  
        }
        
        response = None   
        n_trial = 0       
        
        while n_trial < self.n_trial:
            n_trial += 1
            try:
                api_url = f"https://{self.api_endpoint}/v1/chat/completions"
                res = requests.post(api_url, json=payload, headers=headers, timeout=120)
                
                res.raise_for_status()

                json_data = res.json()
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API (requests) (Attempt {n_trial}/{self.n_trial}): {e}. Retrying...")
                time.sleep(2)
                continue
        
        return response

    def _get_response_httpclient(self, prompt_content: str) -> Optional[str]:
        # Fallback using http.client
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,  
                "messages": [
                    {"role": "user", "content": prompt_content}
                ],
                "temperature": 0.7
            }
        )
        headers = {
            "Authorization": "Bearer " + self.api_key,           
            "Content-Type": "application/json",                  
        }
        
        response = None   
        n_trial = 0       
        
        while n_trial < self.n_trial:
            n_trial += 1
            conn = None
            try:
                # Create HTTPS connection
                conn = http.client.HTTPSConnection(self.api_endpoint, timeout=120)
                # Send POST request
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                # Get response
                res = conn.getresponse()
                data = res.read()
                
                if res.status != 200:
                    raise Exception(f"HTTP Error {res.status}: {res.reason}")

                # Parse JSON response
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API (http.client) (Attempt {n_trial}/{self.n_trial}): {e}. Retrying...")
                time.sleep(2)
                continue
            finally:
                if conn:
                    conn.close()
            
        return response


class InterfaceLLM:
    """Wrapper for InterfaceAPI."""
    def __init__(self, api_endpoint: Optional[str], api_key: Optional[str], model_LLM: Optional[str], debug_mode: bool):
        self.api_endpoint = api_endpoint 
        self.api_key = api_key           
        self.model_LLM = model_LLM       
        self.debug_mode = debug_mode     

        # Check for default/missing settings
        if self.api_key == None or self.api_endpoint ==None or self.api_key == 'xxx' or self.api_endpoint == 'xxx':
            if self.debug_mode:
                print(">> LLM API settings incomplete. LLM functionalities disabled.")
            self.interface_llm = None
            return
        
        # Create InterfaceAPI instance
        self.interface_llm = InterfaceAPI(
            self.api_endpoint,
            self.api_key,
            self.model_LLM,
            self.debug_mode,
        )

    def get_response(self, prompt_content: str) -> Optional[str]:
        if self.interface_llm is None:
             if self.debug_mode:
                print("LLM interface not available. Cannot get response.")
             return None
             
        response = self.interface_llm.get_response(prompt_content)
        print("llm response: ", response)
        return response


# --- Parameter Optimization Module (Integrated) ---

class PARAMETERSEARCH():
    """Automatic parameter optimization using LLM and Optuna."""
    def __init__(self, interface_eval: TSPGLS, interface_llm: InterfaceLLM, paras: Paras):
        if optuna is None:
            self.enabled = False
            return
        
        self.enabled = True
        
        self.interface_llm = interface_llm
        if self.interface_llm is None:
            self.enabled = False
            return

        self.interface_eval = interface_eval
        
        # Settings from Paras
        self.debug_mode = paras.exp_debug_mode
        self.n_trials = paras.opt_n_trials
        self.range_factor = paras.opt_range_factor
        self.storage_path = paras.opt_storage_path
        self.use_numba = paras.eva_numba_decorator
        
        # Define the target function name for TSP
        self.target_func_name = "update_edge_distance"
    
    def construct_prompt_with_emphasis_on_format(self, code_string: str) -> str:
        """Constructs the prompt for the LLM to extract parameters and modify the code for TSP."""
        prompt = (
            "The following Python code defines a heuristic algorithm:\n\n"
            f"```python\n{code_string}\n```\n\n"
            "Your tasks are:\n"
            "1. Identify all tunable numerical parameters (e.g., weights, coefficients, thresholds, factors) that significantly influence the heuristic's decisions. "
            "Return them with their default values in a JSON dictionary.\n"
            f"2. Modify the code to include a new argument, `params`, in the main heuristic function (`{self.target_func_name}`). "
            "All identified tunable parameters must be dynamically retrieved from this `params` dictionary using `.get(key, default_value)`.\n\n"
            "IMPORTANT SAFETY CONSTRAINTS:\n"
            "- Ensure all array dimensions are non-negative\n"
            "- When using numpy.random.choice() or similar sampling functions, ensure sample size <= population size\n"
            "- Validate input parameters to prevent runtime errors\n"
            "- Use appropriate bounds checking for array indices\n"
            "- Handle edge cases gracefully (e.g., empty arrays, zero dimensions)\n\n"
            "The output format **must** strictly follow this structure (do not include any other text):\n\n"
            "1. Extracted Parameters:\n"
            "```json\n"
            "{\n"
            "    \"parameter1_name\": default_value,\n"
            "    \"parameter2_name\": default_value\n"
            "}\n"
            "```\n\n"
            "2. Modified Code:\n"
            "```python\n"
            "def function_name(other_arguments, params):\n"
            "    parameter1_name = params.get(\"parameter1_name\", default_value)\n"
            "    # Function logic...\n"
            "```\n\n"
            "--- Example (TSP Guided Local Search Heuristic) ---\n"
            "Input Code:\n"
            "```python\n"
            "import numpy as np\n"
            "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):\n"
            "    penalty_factor = 0.5\n"
            "    decay_rate = 0.9\n"
            "    adjustment = penalty_factor * edge_n_used\n"
            "    updated_edge_distance = (decay_rate * edge_distance) + adjustment\n"
            "    return updated_edge_distance\n"
            "```\n\n"
            "Expected Output:\n"
            "1. Extracted Parameters:\n"
            "```json\n"
            "{\n"
            "    \"penalty_factor\": 0.5,\n"
            "    \"decay_rate\": 0.9\n"
            "}\n"
            "```\n\n"
            "2. Modified Code:\n"
            "```python\n"
            "import numpy as np\n"
            "def update_edge_distance(edge_distance, local_opt_tour, edge_n_used, params):\n"
            "    penalty_factor = params.get(\"penalty_factor\", 0.5)\n"
            "    decay_rate = params.get(\"decay_rate\", 0.9)\n"
            "    adjustment = penalty_factor * edge_n_used\n"
            "    updated_edge_distance = (decay_rate * edge_distance) + adjustment\n"
            "    return updated_edge_distance\n"
            "```\n"
            "--- End of Example ---\n\n"
            "Now, process the provided input code."
        )
        return prompt

    
    def parse_llm_response(self, response: str) -> Tuple[Dict, Optional[str]]:
        try:
            # Extract parameters section (JSON block)
            # Regex handles optional ```json markdown markers
            params_match = re.search(r"1\.\s*Extracted Parameters:\s*(?:```json)?\s*(\{.*?\})\s*(?:```)?", response, re.DOTALL)
            if params_match:
                params_str = params_match.group(1).strip()
                try:
                    parameters = json.loads(params_str)
                except json.JSONDecodeError:
                    raise ValueError("Failed to decode JSON parameters.")
            else:
                parameters = {}

            # Extract code section (Python block)
            # Requires ```python markdown markers
            code_match = re.search(r"2\.\s*Modified Code:\s*```python\s*(.*?)\s*```", response, re.DOTALL)
            if code_match:
                modified_code = code_match.group(1).strip()
            else:
                # If code is missing but parameters were found, it's an error
                if parameters:
                    if self.debug_mode:
                        print("Warning: Modified code block not found, but parameters were extracted.")
                    return {}, None
                else:
                    # If both are missing, assume no modification needed.
                    return {}, None
            
            # Ensure numpy is imported (essential for TSP)
            modified_code = add_import_package_statement(modified_code, 'numpy', 'np')
            return parameters, modified_code

        except Exception as e:
            if self.debug_mode:
                print(f"[Error] Failed to parse LLM response: {e}")
            return {}, None

    
    def extract_parameters_with_llm(self, code_string: str) -> Tuple[Dict, str]:
        try:
            prompt = self.construct_prompt_with_emphasis_on_format(code_string)
            response = self.interface_llm.get_response(prompt)
            
            if response is None:
                 raise RuntimeError("LLM interface returned None.")

            parameters, modified_code = self.parse_llm_response(response)
            
            # If parsing failed or resulted in empty results, return original
            if modified_code is None:
                 return {}, code_string

            return parameters, modified_code

        except Exception as e:
            if self.debug_mode:
                print(f"[Error] Failed during parameter extraction with LLM: {e}")
            return {}, code_string
        
    def insert_parameters_into_code(self, modified_code: str, best_params: Dict, target_func_name: str) -> str:
        """
        Inserts optimized parameters back into the code by replacing dynamic retrieval 
        with hardcoded values using a robust AST traversal. This version correctly handles
        inline `params.get()` calls within complex expressions.
        """
        if not ast_to_source: 
            if self.debug_mode:
                print("Warning: ast_to_source utility is not available. Cannot insert parameters. Returning parameterized code.")
            return modified_code

        if self.debug_mode:
            print("--- AST Transformation: Inserting hardcoded parameters ---")
            print("Best parameters to insert:", best_params)
            
        try:
            # Parse the parameterized code string into an AST
            tree = ast.parse(modified_code)

            # Define an AST Transformer to find and replace parameter access
            class ParameterInserter(ast.NodeTransformer):
                def __init__(self, params_to_insert, func_name, debug_mode):
                    self.best_params = params_to_insert
                    self.target_func_name = func_name
                    self.debug_mode = debug_mode
                    super().__init__()

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                    if node.name == self.target_func_name:
                        node.args.args = [
                            arg for arg in node.args.args if arg.arg != "params"
                        ]
                        self.generic_visit(node)
                    return node

                def visit_Call(self, node: ast.Call) -> Any:
                    if (
                        isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == "params" and
                        node.func.attr == "get"
                    ):
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            param_key = node.args[0].value
                            if param_key in self.best_params:
                                new_node = ast.Constant(value=self.best_params[param_key])
                                ast.copy_location(new_node, node)
                                if self.debug_mode:
                                    print(f"  -> Replaced params.get('{param_key}') with constant {self.best_params[param_key]}")
                                return new_node
                    
                    self.generic_visit(node)
                    return node

            transformer = ParameterInserter(best_params, target_func_name, self.debug_mode)
            transformed_tree = transformer.visit(tree)
            ast.fix_missing_locations(transformed_tree)

            # Convert the modified AST back to a code string
            optimized_code = ast_to_source(transformed_tree)
            
            # Re-apply Numba decorator if it was specified in the settings
            if self.use_numba:
                optimized_code = add_numba_decorator(optimized_code, target_func_name)

            if self.debug_mode:
                print("--- AST Transformation Complete. Final Optimized Code: ---\n", optimized_code)
                
            return optimized_code

        except Exception as e:
            if self.debug_mode:
                print(f"[Critical Error] Failed to insert parameters into code using AST: {e}")
                traceback.print_exc()
            return modified_code
        
    def get_study_name_from_code(self, code_string: str) -> str:
        # Generate unique hash as the study_name
        return f"study_{hashlib.md5(code_string.encode()).hexdigest()}"

    def get_function_name(self, code_string: str) -> str:
        # Helper to find the main function name
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name == self.target_func_name:
                        return node.name
            # Fallback
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception:
            pass
        return self.target_func_name

    def optimize_and_evaluate(self, code_string: str) -> Tuple[str, float]:
        """Orchestrates the parameter optimization process."""
        if not self.enabled:
            fitness = self.interface_eval.evaluate(code_string, n_evals=3)
            return code_string, fitness if fitness is not None else float("inf")

        func_name = self.get_function_name(code_string)
        
        try:
            # 1. Call LLM to extract parameters and generate modified code
            parameters, modified_code = self.extract_parameters_with_llm(code_string)

            if not parameters or modified_code == code_string or modified_code is None:
                if self.debug_mode:
                    print("[Info] No parameters extracted. Skipping optimization.")
                fitness = self.interface_eval.evaluate(code_string, n_evals=3)
                return code_string, fitness if fitness is not None else float("inf")
            
            # 2. Process parameters: Convert defaults to search ranges
            parameter_ranges = {}
            for key, val in parameters.items():
                # Ensure value is numerical
                if not isinstance(val, (int, float)):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        if self.debug_mode:
                            print(f"  -> Skipping non-numeric parameter '{key}'.")
                        continue # Skip non-numeric parameters
                
                val = float(val)

                if abs(val) < 1e-9:
                    low = -self.range_factor
                    high = self.range_factor
                    if self.debug_mode:
                        print(f"  -> Parameter '{key}' is zero. Using symmetric range: ({low:.2f}, {high:.2f})")
                elif abs(val) < 1e-4:
                    low = val - 0.1
                    high = val + 0.1
                    if self.debug_mode:
                        print(f"  -> Parameter '{key}' is very small. Using fixed additive range: ({low:.2f}, {high:.2f})")
                else:
                    range_size = abs(val) * 0.1  # +/- 50% of the default value
                    low = val - range_size
                    high = val + range_size
                    if self.debug_mode:
                        print(f"  -> Parameter '{key}' default is {val:.4f}. Using percentage range: ({low:.4f}, {high:.4f})")
                
                parameter_ranges[key] = (low, high)

            if not parameter_ranges:
                if self.debug_mode:
                    print("[Info] No valid numeric parameters found to optimize.")
                fitness = self.interface_eval.evaluate(code_string, n_evals=3)
                return code_string, fitness if fitness is not None else float("inf")

            # 3. Define Optuna objective function
            def objective(trial):
                params = {}
                for key, (low, high) in parameter_ranges.items():
                    if low > high:
                        low, high = high, low
                    params[key] = trial.suggest_float(key, low, high)
                
                # Ensure Numba is applied during optimization if required
                code_to_eval = modified_code
                if self.use_numba:
                     try:
                         code_to_eval = add_numba_decorator(modified_code, func_name)
                     except Exception:
                         pass

                fitness = self.interface_eval.evaluate(code_to_eval, params=params, n_evals=1)
                
                if fitness is None:
                    raise optuna.TrialPruned()
                    
                return fitness

            # 4. Setup and run Optuna optimization
            study_name = self.get_study_name_from_code(modified_code)
            
            # Configure storage
            try:
                storage = RDBStorage(
                    url=self.storage_path,
                    engine_kwargs={"connect_args": {"timeout": 30.0}}
                )
                try:
                    optuna.delete_study(study_name=study_name, storage=storage)
                except (KeyError, Exception):
                    pass
            except Exception as e:
                if self.debug_mode:
                    print(f"[Warning] RDBStorage failed: {e}. Falling back to InMemoryStorage.")
                storage = InMemoryStorage()

            study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=False)

            if not self.debug_mode:
                optuna.logging.set_verbosity(optuna.logging.WARNING)

            study.optimize(objective, n_trials=self.n_trials, n_jobs=1, gc_after_trial=True)

            if len(study.trials) == 0 or study.best_trial is None or study.best_trial.state != optuna.trial.TrialState.COMPLETE:
                if self.debug_mode:
                    print("[Warning] Optimization completed with no successful trials.")
                fitness = self.interface_eval.evaluate(code_string, n_evals=3)
                return code_string, fitness if fitness is not None else float("inf")

            best_params = study.best_params

            # 5. Insert the best parameters back into the code (hardcoding)
            optimized_code = self.insert_parameters_into_code(modified_code, best_params, func_name)

            # 6. Final evaluation of the optimized code (n_evals=5 for robustness)
            final_fitness = self.interface_eval.evaluate(optimized_code, n_evals=5)
            
            if final_fitness is None:
                final_fitness = float("inf")

            return optimized_code, final_fitness

        except Exception as e:
            if self.debug_mode:
                print(f"[Error] Failed during optimization process: {e}")
                traceback.print_exc()
            
            try:
                fitness = self.interface_eval.evaluate(code_string, n_evals=3)
                return code_string, fitness if fitness is not None else float("inf")
            except Exception:
                 return code_string, float("inf")


# --- Evolution Components (Meta-Level: Prompts) ---

class Evolution_Prompt():
    """Handles the evolution of the prompts themselves (Meta-Heuristics)."""

    def __init__(self, interface_llm: InterfaceLLM, debug_mode: bool, problem_type: str, **kwargs):
        self.prompt_task = f"We are working on solving a {problem_type} problem." + \
        " Our objective is to leverage the capabilities of the Language Model (LLM) to generate heuristic algorithms." + \
        " We need your assistance in analyzing existing prompts and their results to generate new prompts that will help us achieve better outcomes."

        self.debug_mode = debug_mode          
        self.interface_llm = interface_llm
    
        
    def get_prompt_cross(self,prompts_indivs: List[Dict]) -> str:
        # Combine prompts and their performance metrics
        prompt_indiv = ""
        for i in range(len(prompts_indivs)):
            obj_val = prompts_indivs[i]['objective']
            obj_str = f"{obj_val:.5f}" if isinstance(obj_val, float) else str(obj_val)
            prompt_indiv += f"No.{i+1} prompt's tasks assigned to LLM, and objective function value are: \n{prompts_indivs[i]['prompt']}\nObjective: {obj_str}\n\n"
        
        # Request a new, different prompt
        prompt_content = self.prompt_task+"\n"\
f"I have {len(prompts_indivs)} existing prompt(s) with objective function value(s) as follows: \n"\
+prompt_indiv+\
"Please help me create a new prompt that has a totally different form from the given ones but can be motivated from them. \n" +\
"Please describe your new prompt and main steps in one sentence."\
+"\n"+"Do not give additional explanations. Just the sentence."
        return prompt_content
    
    
    def get_prompt_variation(self,prompts_indivs: List[Dict]) -> str:
        # Request a modified version of the provided prompt
        obj_val = prompts_indivs[0]['objective']
        obj_str = f"{obj_val:.5f}" if isinstance(obj_val, float) else str(obj_val)

        prompt_content = self.prompt_task+"\n"\
"I have one prompt with its objective function value as follows." + \
"prompt description: " + prompts_indivs[0]['prompt'] + "\n" + \
f"objective function value: {obj_str}\n\n" +\
"Please assist me in creating a new prompt that has a different form but can be a modified version of the algorithm provided. \n" +\
"Please describe your new prompt and main steps in one sentence." \
+"\n"+"Do not give additional explanations. Just the sentence."
        return prompt_content
    
    def initialize(self, prompt_type: str) -> List[str]:
        # Initialize base prompts (from original TSP script)
        if(prompt_type == 'cross'):
            prompt_content = ['Please help me create a new algorithm that has a totally different form from the given ones.', \
                              'Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.']
        else:
            prompt_content = ['Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.', \
                              'Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.']            
        return prompt_content
    
    def cross(self,parents: List[Dict]) -> Optional[str]:
        if self.interface_llm is None: return None
        prompt_content = self.get_prompt_cross(parents)
        response = self.interface_llm.get_response(prompt_content)
        return response
    
    def variation(self,parents: List[Dict]) -> Optional[str]:
        if self.interface_llm is None: return None
        prompt_content = self.get_prompt_variation(parents)
        response = self.interface_llm.get_response(prompt_content)
        return response

class InterfaceEC_Prompt():
    """Interface for Evolving Prompts."""
    def __init__(self, pop_size: int, m: int, interface_llm: InterfaceLLM, debug_mode: bool, select, n_p: int, timeout: int, problem_type: str, **kwargs):
        self.pop_size = pop_size
        self.evol = Evolution_Prompt(interface_llm, debug_mode, problem_type , **kwargs)
        self.m = m
        self.debug = debug_mode
        if not self.debug:
            warnings.filterwarnings("ignore")
        self.select = select
        self.n_p = n_p
        self.timeout = timeout

    def extract_prompt_text(self, text: Optional[str]) -> str:
        # Robust extraction helper
        if text is None:
            return ""
        
        # 1. Try extracting from double quotes
        match = re.search(r'"(.*?)"', text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
        else:
            extracted = text.strip()

        # 2. Remove potential prefix "Prompt: "
        prefix = "Prompt: "
        if extracted.startswith(prefix):
            return extracted[len(prefix):].strip()
        
        # 3. Remove braces if LLM wrapped the prompt in them
        if extracted.startswith("{") and extracted.endswith("}"):
             return extracted[1:-1].strip()
             
        return extracted
    
    
    # Generate offspring based on the specified evolution operator
    def _get_alg(self,pop: List[Dict], operator: str):
        DEFAULT_OBJ = 1e9
        offspring = {
            'prompt': None,
            'objective': DEFAULT_OBJ, 
            'number': []      # Performance tracking history
        }
        off_set = []
        parents = []

        # Get initial prompts
        if operator == "initial_cross":
            prompt_list =  self.evol.initialize("cross")
            for prompt in prompt_list:
                off_set.append({'prompt': prompt, 'objective': DEFAULT_OBJ, 'number': []})
        elif operator == "initial_variation":
            prompt_list =  self.evol.initialize("variation")   
            for prompt in prompt_list:
                off_set.append({'prompt': prompt, 'objective': DEFAULT_OBJ, 'number': []})
        # Generate new prompt via crossover      
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            prompt_now = self.evol.cross(parents)
            prompt_new = self.extract_prompt_text(prompt_now)
            offspring["prompt"] = prompt_new

        # Generate new prompt via variation
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            prompt_now = self.evol.variation(parents)
            prompt_new = self.extract_prompt_text(prompt_now)
            offspring["prompt"] = prompt_new

        elif operator not in ["initial_cross", "initial_variation"]:
            print(f"Prompt operator [{operator}] has not been implemented ! \n") 

        return parents, offspring, off_set

    # Generate offspring (wrapper for _get_alg)
    def get_offspring(self, pop: List[Dict], operator: str):
        try:
            p, offspring, off_set = self._get_alg(pop, operator)
        except Exception as e:
            if self.debug:
                print("get_offspring (Prompt) error:", e)
                traceback.print_exc()
            offspring = {'prompt': None, 'objective': 1e9, 'number': []}
            p = None
            off_set = None
        return p, offspring, off_set
    
    def get_algorithm(self, pop: List[Dict], operator: str):
        results = []
        n_jobs = min(self.n_p, self.pop_size) if operator in ['cross', 'variation'] else 1
        
        try:
            # Timeout increased (+60s) for LLM latency
            if(operator == 'cross' or operator == 'variation'):
                results = Parallel(n_jobs=n_jobs, timeout=self.timeout+60)(delayed(self.get_offspring)(pop, operator) for _ in range(self.pop_size))
            else:
                results = Parallel(n_jobs=n_jobs, timeout=self.timeout+60)(delayed(self.get_offspring)(pop, operator) for _ in range(1))
        except Exception as e:
            if self.debug:
                print(f"Error in Parallel execution (Prompt): {e}")
            
        time.sleep(1)

        out_p = []
        out_off = []

        for p, off, off_set in results:
            out_p.append(p)
            if(operator == 'cross' or operator == 'variation'):
                if off and off['prompt']:
                    out_off.append(off)
            else:
                if off_set:
                    for now_off in off_set:
                        if now_off and now_off['prompt']:
                             out_off.append(now_off)
        return out_p, out_off

    def population_generation(self, initial_type: str) -> List[Dict]:
        n_create = 1
        population = []
        for i in range(n_create):
            _,pop = self.get_algorithm([], initial_type)
            for p in pop:
                population.append(p)
        return population

    
# --- Evolution Components (Base-Level: Heuristics) ---

class Evolution():
    """Handles the generation of new heuristic algorithms using LLM."""

    def __init__(self, interface_llm: InterfaceLLM, debug_mode: bool, prompts: GetPrompts, **kwargs):
        self.prompt_task         = prompts.get_task()         
        self.prompt_func_name    = prompts.get_func_name()    # "update_edge_distance"
        self.prompt_func_inputs  = prompts.get_func_inputs()  
        self.prompt_func_outputs = prompts.get_func_outputs() 
        self.prompt_inout_inf    = prompts.get_inout_inf()    
        self.prompt_other_inf    = prompts.get_other_inf()    
        
        # Format inputs/outputs for the prompt string
        self.joined_inputs = ", ".join(f"'{s}'" for s in self.prompt_func_inputs)
        self.joined_outputs = ", ".join(f"'{s}'" for s in self.prompt_func_outputs)

        self.debug_mode = debug_mode          
        self.interface_llm = interface_llm

    def get_prompt_initial(self) -> str:
        # Construct the prompt content with markdown block instructions
        prompt_content = self.prompt_task+"\n"\
"Please design a creative and innovative algorithm. "\
"The output format must strictly follow this structure:\n\n"\
"First, provide a one-sentence description of your algorithm enclosed in curly braces {}. "\
"Then, implement it in Python as a function named "\
+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"\
"IMPORTANT REQUIREMENTS:\n"\
+"- Focus on algorithm innovation, do not add input validation or error checking code\n"\
+"- Ensure the code is syntactically correct and executable\n"\
+"- The description in {} should be placed BEFORE the Python code block\n"\
+"- Do not include any robustness checks or defensive programming patterns\n"\
+"- Keep the implementation clean and focused on the core algorithm logic\n"\
"Output format:\n"\
"{Your algorithm description here}\n\n"\
"```python\n"\
"# Your Python implementation here\n"\
"```\n\n"\
"Do not give additional explanations outside this format."
        return prompt_content

        
    def get_prompt_cross(self,indivs: List[Dict], prompt: str) -> str:
        # Combine thoughts, objectives, and codes of parents
        prompt_indiv = ""
        for i in range(len(indivs)):
            obj_val = indivs[i]['objective']
            obj_str = f"{obj_val:.5f}" if isinstance(obj_val, float) else str(obj_val)
            prompt_indiv += f"No.{i+1} algorithm's thought, objective function value, and the corresponding code are: \nThought: {indivs[i]['algorithm']}\nObjective: {obj_str}\nCode:\n```python\n{indivs[i]['code']}\n```\n\n"
        
        # Construct the prompt content using the meta-prompt
        prompt_content = self.prompt_task+"\n"\
f"I have {len(indivs)} existing algorithm(s) with their details as follows: \n"\
+prompt_indiv+ "Instruction for the new algorithm: " + prompt + "\n\n" +\
"Please design a creative and innovative algorithm that combines or improves upon the existing ones. "\
"The output format must strictly follow this structure:\n\n"\
"First, provide a one-sentence description of your algorithm enclosed in curly braces {}. "\
"Then, implement it in Python as a function named "\
+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"\
"IMPORTANT REQUIREMENTS:\n"\
+"- Focus on algorithm innovation, do not add input validation or error checking code\n"\
+"- Ensure the code is syntactically correct and executable\n"\
+"- The description in {} should be placed BEFORE the Python code block\n"\
+"- Do not include any robustness checks or defensive programming patterns\n"\
+"- Keep the implementation clean and focused on the core algorithm logic\n"\
"Output format:\n"\
"{Your algorithm description here}\n\n"\
"```python\n"\
"# Your Python implementation here\n"\
"```\n\n"\
"Do not give additional explanations outside this format."
        return prompt_content
    
    
    def get_prompt_variation(self,indiv1: Dict, prompt: str) -> str:
        # Construct the prompt content using the meta-prompt
        prompt_content = self.prompt_task+"\n"\
"I have one algorithm with its code as follows. \
Algorithm description: "+indiv1['algorithm']+"\n\
Code:\n\
```python\n"+indiv1['code']+"\n```\n\n" + \
"Instruction for the new algorithm: " + prompt + "\n\n" + \
"Please design a creative and innovative algorithm that varies or improves upon the existing one. "\
"The output format must strictly follow this structure:\n\n"\
"First, provide a one-sentence description of your algorithm enclosed in curly braces {}. "\
"Then, implement it in Python as a function named "\
+self.prompt_func_name +". This function should accept "+str(len(self.prompt_func_inputs))+" input(s): "\
+self.joined_inputs+". The function should return "+str(len(self.prompt_func_outputs))+" output(s): "\
+self.joined_outputs+". "+self.prompt_inout_inf+" "\
+self.prompt_other_inf+"\n"\
"IMPORTANT REQUIREMENTS:\n"\
+"- Focus on algorithm innovation, do not add input validation or error checking code\n"\
+"- Ensure the code is syntactically correct and executable\n"\
+"- The description in {} should be placed BEFORE the Python code block\n"\
+"- Do not include any robustness checks or defensive programming patterns\n"\
+"- Keep the implementation clean and focused on the core algorithm logic\n"\
"Output format:\n"\
"{Your algorithm description here}\n\n"\
"```python\n"\
"# Your Python implementation here\n"\
"```\n\n"\
"Do not give additional explanations outside this format."
        return prompt_content
    

    def _get_alg(self,prompt_content: str) -> Tuple[Optional[str], Optional[str]]:
        if self.interface_llm is None:
             return None, None

        response = self.interface_llm.get_response(prompt_content)
        if response is None:
             return None, None

        # Function to extract algorithm description and code from response
        def extract(response_text):
            # Extract description (inside braces)
            alg = re.findall(r"\{(.*)\}", response_text, re.DOTALL)
            
            # Extract code block (inside ```python ... ```)
            code_block = re.findall(r"```python\n(.*?)\n```", response_text, re.DOTALL)
            
            # Fallback extraction patterns
            if len(alg) == 0:
                if code_block:
                    # Extract everything before the code block starts
                    alg_match = re.search(r'^(.*?)```python', response_text, re.DOTALL)
                    if alg_match: alg = [alg_match.group(1).strip()]
                elif 'import' in response_text:
                    alg = re.findall(r'^.*?(?=import)', response_text,re.DOTALL)
                elif 'def' in response_text:
                    alg = re.findall(r'^.*?(?=def)', response_text,re.DOTALL)

            if code_block:
                c = [code_block[0]]
            else:
                # Fallback if LLM provided raw code
                c = re.findall(r"((?:import\s+.*?\n)*\s*def\s+\w+\s*\(.*?\):.*)", response_text, re.DOTALL)
            
            return alg, c

        algorithm, code = extract(response)

        # Retry mechanism
        n_retry = 1
        MAX_RETRIES = 3
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print(f"Error: algorithm or code not identified (Retry {n_retry}/{MAX_RETRIES}). Retrying...")
            time.sleep(1)

            response = self.interface_llm.get_response(prompt_content)
            if response is None:
                 break

            algorithm, code = extract(response)
            
            if n_retry >= MAX_RETRIES:
                break
            n_retry +=1

        if len(algorithm) == 0 or len(code) == 0:
            if self.debug_mode:
                 print("Failed to extract algorithm or code after retries.")
            return None, None

        # Extract the first match and strip whitespace
        algorithm_text = algorithm[0].strip()
        code_all = code[0].strip() 

        return code_all, algorithm_text


    def initial(self) -> Tuple[Optional[str], Optional[str]]:
        prompt_content = self.get_prompt_initial()
        return self._get_alg(prompt_content)
    
    def cross(self, parents: List[Dict], prompt: str) -> Tuple[Optional[str], Optional[str]]:
        prompt_content = self.get_prompt_cross(parents, prompt)
        return self._get_alg(prompt_content)
    
    def variation(self,parents: Dict, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        prompt_content = self.get_prompt_variation(parents, prompt)
        return self._get_alg(prompt_content)
    

class InterfaceEC():
    """Interface for Evolving Heuristics (Base-Level)."""
    def __init__(self, pop_size: int, m: int, interface_llm: InterfaceLLM, debug_mode: bool, interface_prob: TSPGLS, select, n_p: int, timeout: int, use_numba: bool, **kwargs):
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(interface_llm, debug_mode, prompts, **kwargs)
        self.m = m
        self.debug = debug_mode
        if not self.debug:
            warnings.filterwarnings("ignore")
        self.select = select
        self.n_p = n_p
        self.timeout = timeout
        self.use_numba = use_numba
    
    def check_duplicate(self,population: List[Dict], code: str) -> bool:
        for ind in population:
            if code == ind['code']:
                return True
        return False
    
    # Generate offspring based on the specified operator
    def _get_alg(self,pop: List[Dict], operator: str, prompt: str):
        offspring = {
            'algorithm': None, 'code': None, 'objective': None, 'other_inf': None
        }
        # Get initial algorithm
        if operator == "initial":
            parents = None
            offspring['code'], offspring['algorithm'] = self.evol.initial()    
        # Crossover        
        elif operator == "cross":
            parents = self.select.parent_selection(pop,self.m)
            offspring['code'], offspring['algorithm'] = self.evol.cross(parents, prompt)
        # Variation
        elif operator == "variation":
            parents = self.select.parent_selection(pop,1)
            offspring['code'], offspring['algorithm'] = self.evol.variation(parents[0], prompt)    
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring

    # Generate offspring and evaluate fitness
    def get_offspring(self, pop: List[Dict], operator: str, prompt: str):
        try:
            p, offspring = self._get_alg(pop, operator, prompt)
            
            if offspring['code'] is None:
                 raise ValueError("LLM failed to generate code.")
            
            # Preprocessing: Ensure necessary imports and Numba acceleration
            code = offspring['code']
            
            # Ensure numpy is imported (Crucial for TSP)
            code = add_import_package_statement(code, 'numpy', 'np')
            
            function_name = None
            if self.use_numba:
                try:
                    # Extract function name using regex
                    pattern = r"def\s+(\w+)\s*\(.*\):"
                    match = re.search(pattern, code)
                    if match:
                        function_name = match.group(1)
                        code = add_numba_decorator(program=code, function_name=function_name)
                except Exception as e:
                    if self.debug:
                        print(f"Numba decorator addition failed: {e}.")

            # Handle duplicate code
            n_retry= 0
            MAX_RETRIES = 1
            while self.check_duplicate(pop, code):
                n_retry += 1
                if n_retry > MAX_RETRIES:
                    raise ValueError("Duplicated code after retry.")
                
                if self.debug:
                    print(f"Duplicated code detected. Retrying...")
                time.sleep(1)
                
                p, offspring = self._get_alg(pop, operator, prompt)
                if offspring['code'] is None:
                    raise ValueError("LLM failed to generate code on retry.")

                code = offspring['code']
                code = add_import_package_statement(code, 'numpy', 'np')
                if self.use_numba and function_name:
                    try:
                        code = add_numba_decorator(program=code, function_name=function_name)
                    except Exception:
                         pass
                
            # Update offspring code after preprocessing
            offspring['code'] = code

            # Evaluate fitness using ThreadPoolExecutor for timeout management
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.interface_eval.evaluate, code, n_evals=1)
                try:
                    fitness = future.result(timeout=self.timeout)
                except concurrent.futures.TimeoutError:
                    fitness = None
                
                if fitness is not None:
                    offspring['objective'] = np.round(fitness, 5)
                else:
                    raise ValueError("Evaluation failed or returned None.")
                    
        # Handle exceptions
        except Exception as e:
            if self.debug:
                print(f"Error in get_offspring: {e}")
            offspring = {
                'algorithm': None, 'code': None, 'objective': None, 'other_inf': None
            }
            p = None

        return p, offspring
    
    def get_algorithm(self, pop: List[Dict], operator: str, prompt: str):
        results = []
        n_jobs = min(self.n_p, self.pop_size)
        
        try:
            # Timeout increased (+60s) for overhead
            results = Parallel(n_jobs=n_jobs, timeout=self.timeout+60)(delayed(self.get_offspring)(pop, operator, prompt) for _ in range(self.pop_size))
        except Exception as e:
            if self.debug:
                print(f"Error in Parallel execution (Algorithm): {e}")
            
        time.sleep(1)

        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            # Only add valid offspring
            if off and off['objective'] is not None:
                out_off.append(off)
        return out_p, out_off

    def population_generation(self) -> List[Dict]:
        # Generate initial population (2 rounds as in original script)
        n_create = 2
        population = []
        for i in range(n_create):
            _,pop = self.get_algorithm([],'initial', [])
            for p in pop:
                population.append(p)
        return population


# --- EOH Framework Runner ---

class EOH:
    def __init__(self, paras: Paras, problem: TSPGLS, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # EoH Prompt Evolution settings (Meta-level)
        self.pop_size_cross = 2
        self.pop_size_variation = 2
        self.problem_type = "minimization"

        # Experimental settings (Base-level)      
        self.pop_size = paras.ec_pop_size
        self.n_pop = paras.ec_n_pop
        self.operators = paras.ec_operators
        
        self.operator_weights = paras.ec_operator_weights    
        if (paras.ec_m > self.pop_size or paras.ec_m < 2) and self.pop_size >= 2:
            print(f"Warning: ec_m ({paras.ec_m}) adjusted to m=2.")
            paras.ec_m = 2
        self.m = paras.ec_m
        self.m_prompt = 2

        self.debug_mode = paras.exp_debug_mode
        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator
        
        self.paras = paras

        print("- EoH parameters loaded -")

        # Initialize Central LLM Interface
        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.llm_model, self.debug_mode)

        # Ensure evaluator debug mode matches
        if hasattr(self.prob, 'debug_mode'):
            self.prob.debug_mode = self.debug_mode

        # Initialize Parameter Optimization Interface (Integrated Component)
        self.parameter_search = PARAMETERSEARCH(
            interface_eval=self.prob,
            interface_llm=self.interface_llm,
            paras=self.paras
        )

        # Set random seed
        random.seed(2024)

    # Add offspring to the population (Heuristics)
    def add2pop(self, population: List[Dict], offsprings: List[Dict]):
        for off in offsprings:
            if off['objective'] is None:
                continue
                
            is_duplicate = False
            # In debug mode, check for objective duplication
            if self.debug_mode:
                 for ind in population:
                    # Use a small tolerance for float comparison
                    if abs(ind['objective'] - off['objective']) < 1e-9:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                population.append(off)
    
    # Add offspring to the population (Prompts)
    def add2pop_prompt(self, population: List[Dict], offsprings: List[Dict]):
        for off in offsprings:
            if not off or not off['prompt']:
                continue

            is_duplicate = False
            # Check for duplicates based on prompt text
            for ind in population:
                if ind['prompt'] == off['prompt']:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                 population.append(off)
    

    def optimize_population_parameters(self, population: List[Dict], generation: int = -1) -> List[Dict]:
        # (Requirement R4)
        print(f"\n=== Starting Parameter Optimization Phase ===")
        
        valid_individuals = [ind for ind in population if ind['objective'] is not None and ind['code'] is not None]
        
        if not valid_individuals:
            print("No valid individuals to optimize.")
            return population

        new_population = []
        
        # Optimization is done sequentially
        for i, indiv in enumerate(valid_individuals):
            print(f"Optimizing individual {i+1}/{len(valid_individuals)} (Current Obj: {indiv['objective']:.5f})...")
            
            # Run optimization
            optimized_code, final_fitness = self.parameter_search.optimize_and_evaluate(indiv['code'])

            # Check if optimization improved the result (lower fitness is better)
            if final_fitness < indiv['objective'] - 1e-9 and np.isfinite(final_fitness):
                print(f"  -> Optimization SUCCESS. Improvement: {indiv['objective'] - final_fitness:.5f}. New Obj: {final_fitness:.5f}")
                # Create updated individual
                new_indiv = {
                    'algorithm': indiv['algorithm'] + " [Optimized]",
                    'code': optimized_code,
                    'objective': np.round(final_fitness, 5),
                    'other_inf': "Parameters Optimized"
                }
                new_population.append(new_indiv)
            else:
                if self.debug_mode:
                    if not np.isfinite(final_fitness):
                        print("  -> Optimization FAILED (Evaluation error/timeout). Keeping original.")
                    else:
                        print(f"  -> Optimization finished, but no significant improvement found (New Obj: {final_fitness:.5f}). Keeping original.")
                new_population.append(indiv)

            # Save intermediate population after each individual optimization (if generation is provided)
            if generation > 0:
                # Add back any individuals that were initially invalid so far
                current_invalid = [ind for ind in population if ind not in valid_individuals[:i+1]]
                current_population = new_population + current_invalid
                
                # Apply current population management
                current_managed_population = self.manage.population_management(current_population, self.pop_size)
                
                # Save intermediate state
                filename_intermediate = os.path.join(self.output_path, f"results/pops/population_generation_{generation}_param_opt_step_{i+1}.json")
                try:
                    with open(filename_intermediate, 'w') as f:
                        json.dump(current_managed_population, f, indent=5)
                    print(f"  -> Saved intermediate population after optimizing individual {i+1}: {filename_intermediate}")
                except IOError as e:
                    print(f"  -> Error saving intermediate population: {e}")
                
                # Save intermediate best individual
                if current_managed_population:
                    filename_best_intermediate = os.path.join(self.output_path, f"results/pops_best/population_generation_{generation}_param_opt_step_{i+1}.json")
                    try:
                        with open(filename_best_intermediate, 'w') as f:
                            json.dump(current_managed_population[0], f, indent=5)
                    except IOError:
                        pass

        # Add back any individuals that were initially invalid
        invalid_individuals = [ind for ind in population if ind not in valid_individuals]
        new_population.extend(invalid_individuals)

        # Re-sort and manage the population after optimization
        final_population = self.manage.population_management(new_population, self.pop_size)
        print("=== Parameter Optimization Phase Finished ===")
        return final_population


    # Run the EOH process
    def run(self):
        print("- Evolution Start -")
        time_start = time.time()

        interface_prob = self.prob
        
        # Initialize Prompt Evolution Interfaces (Meta-Level)
        interface_promt_cross = InterfaceEC_Prompt(self.pop_size_cross, self.m_prompt, self.interface_llm, self.debug_mode, self.select, self.exp_n_proc, self.timeout, self.problem_type)
        interface_promt_variation = InterfaceEC_Prompt(self.pop_size_variation, self.m_prompt, self.interface_llm, self.debug_mode, self.select, self.exp_n_proc, self.timeout, self.problem_type)
        
        # Initialize Heuristic Evolution Interface (Base-Level)
        interface_ec = InterfaceEC(self.pop_size, self.m, self.interface_llm,
                                   self.debug_mode, interface_prob, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba
                                   )

        # Initialize Prompt Populations
        print("Creating initial prompts:")
        cross_operators = interface_promt_cross.population_generation("initial_cross")
        variation_operators = interface_promt_variation.population_generation("initial_variation")

        print("=======================================")
        # Initialize Heuristic Population
        print("Creating initial population (heuristics):")
        population = interface_ec.population_generation()
        
        # Filter out failed initializations
        population = [ind for ind in population if ind['objective'] is not None]

        if not population:
             print("Warning: Initial population generation resulted in 0 valid heuristics.")

        # Apply population management
        if population:
            population = self.manage.population_management(population, self.pop_size)

        # NOTE: Parameter optimization no longer applied to initial population.
        # It will now run ONLY once at the final generation to save time and
        # focus evaluation budget on evolved heuristics.

        print(f"Pop initial (after optimization): ")
        if population:
            for off in population:
                print(f" Obj: {off['objective']:.5f}", end="|")
        print("\nInitial population created and optimized!")
        
        # Save initial population
        filename = os.path.join(self.output_path, "results/pops/population_generation_0.json")
        try:
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
        except IOError as e:
            print(f"Error saving initial population: {e}")

        n_start = 0

        print("=======================================")

        # --- Main Evolution Loop ---
        
        # Variables for adaptive prompt evolution strategy (from original TSP script)
        worst_objectives = [] 
        delay_turn = 3        
        change_flag = 0       
        last_prompt_evolution_gen = -1 
        max_k = 4             # History size for prompt performance tracking

        for gen in range(n_start, self.n_pop):  
            
            # --- Adaptive Prompt Evolution Logic (Meta-Level) ---
            
            # If cooldown is active, finalize the prompt population update.
            if(change_flag):
                change_flag -= 1
                if(change_flag == 0):
                    print("--- Finalizing Prompt Population Update (Sorting/Selection) ---")
                    cross_operators = self.manage.population_management(cross_operators, self.pop_size_cross)
                    variation_operators = self.manage.population_management(variation_operators, self.pop_size_variation)

            # Check for stagnation
            if(len(worst_objectives) >= delay_turn and worst_objectives[-1] >= worst_objectives[-delay_turn] - 1e-9 and gen - last_prompt_evolution_gen > delay_turn):
                print(f"--- Stagnation detected. Evolving Prompts (Generation {gen+1}) ---")
                
                # Evolve Cross Prompts
                _, offsprings_c_cross = interface_promt_cross.get_algorithm(cross_operators, 'cross')
                self.add2pop_prompt(cross_operators, offsprings_c_cross)
                _, offsprings_c_var = interface_promt_cross.get_algorithm(cross_operators, 'variation')
                self.add2pop_prompt(cross_operators, offsprings_c_var)
                
                # Reset metrics for newly added prompts (identified by 1e9 default)
                for prompt in cross_operators:
                    if prompt["objective"] == 1e9:
                         prompt["number"] = []

                # Evolve Variation Prompts
                _, offsprings_v_cross = interface_promt_variation.get_algorithm(variation_operators, 'cross')
                self.add2pop_prompt(variation_operators, offsprings_v_cross)
                _, offsprings_v_var = interface_promt_variation.get_algorithm(variation_operators, 'variation')
                self.add2pop_prompt(variation_operators, offsprings_v_var)

                for prompt in variation_operators:
                     if prompt["objective"] == 1e9:
                        prompt["number"] = []
                
                change_flag = 2 # Set cooldown
                last_prompt_evolution_gen = gen

            # --- Heuristic Evolution Phase (Base-Level) ---
            print(f"--- Generation {gen+1}: Heuristic Evolution ---")

            # 1. Crossover operations
            for i in range(len(cross_operators)):
                prompt_text = cross_operators[i]["prompt"]
                print(f" OP: cross, Prompt [{i + 1}/{len(cross_operators)}]", end=" | Offspring: ") 
                
                parents, offsprings = interface_ec.get_algorithm(population, "cross", prompt_text)
                self.add2pop(population, offsprings)  
                
                # Update prompt performance metrics
                for off in offsprings:
                    print(f"{off['objective']:.5f}", end="|")

                    # Track performance using a max-heap (storing negative objectives)
                    if len(cross_operators[i]["number"]) < max_k:
                        heapq.heappush(cross_operators[i]["number"], -off['objective'])
                    else:
                        # If new objective is better (smaller) than the worst in the heap
                        if off['objective'] < -cross_operators[i]["number"][0]:
                            heapq.heapreplace(cross_operators[i]["number"], -off['objective'])  
                        
                    # Prompt's objective is the average of its top 'max_k' performances
                    if cross_operators[i]["number"]:
                        cross_operators[i]["objective"] = -sum(cross_operators[i]["number"]) / len(cross_operators[i]["number"])
                
                # Apply population management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f" | Prompt Fitness: {cross_operators[i]['objective']:.5f}")

            
            # 2. Variation operations
            for i in range(len(variation_operators)):
                prompt_text = variation_operators[i]["prompt"]
                print(f" OP: variation, Prompt [{i + 1}/{len(variation_operators)}]", end=" | Offspring: ") 
                
                parents, offsprings = interface_ec.get_algorithm(population, "variation", prompt_text)
                self.add2pop(population, offsprings)  
                
                # Update prompt performance metrics
                for off in offsprings:
                    print(f"{off['objective']:.5f}", end="|")

                    if len(variation_operators[i]["number"]) < max_k:
                        heapq.heappush(variation_operators[i]["number"], -off['objective'])
                    else:
                        if off['objective'] < -variation_operators[i]["number"][0]:
                            heapq.heapreplace(variation_operators[i]["number"], -off['objective'])
                    
                    if variation_operators[i]["number"]:
                        variation_operators[i]["objective"] = -sum(variation_operators[i]["number"]) / len(variation_operators[i]["number"])
                
                # Apply population management
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print(f" | Prompt Fitness: {variation_operators[i]['objective']:.5f}")


            # --- Parameter Optimization Phase (Final Generation Only) ---
            if population and gen == self.n_pop - 1:
                print("\n>>> Final generation reached: executing parameter optimization on surviving population (single run).")
                
                # Save population BEFORE parameter optimization
                filename_before = os.path.join(self.output_path, "results/pops/population_generation_" + str(gen + 1) + "_before_param_opt.json")
                try:
                    with open(filename_before, 'w') as f:
                        json.dump(population, f, indent=5)
                    print(f"Saved population before parameter optimization: {filename_before}")
                except IOError as e:
                    print(f"Error saving population before optimization: {e}")
                
                # Save best individual BEFORE parameter optimization
                if population:
                    filename_best_before = os.path.join(self.output_path, "results/pops_best/population_generation_" + str(gen + 1) + "_before_param_opt.json")
                    try:
                        with open(filename_best_before, 'w') as f:
                            json.dump(population[0], f, indent=5)
                    except IOError:
                        pass
                
                # Run parameter optimization with intermediate recording
                population = self.optimize_population_parameters(population, gen + 1)
            elif self.debug_mode and gen == 0:
                print(">>> Skipping parameter optimization for intermediate generations (will run only at final generation).")

            # --- End of Generation ---

            # Save results
            if gen == self.n_pop - 1 and population:
                # For final generation, save the post-optimization population with special naming
                filename = os.path.join(self.output_path, "results/pops/population_generation_" + str(gen + 1) + "_after_param_opt.json")
                try:
                    with open(filename, 'w') as f:
                        json.dump(population, f, indent=5)
                    print(f"Saved final population after parameter optimization: {filename}")
                except IOError:
                    pass

                filename = os.path.join(self.output_path, "results/pops_best/population_generation_" + str(gen + 1) + "_after_param_opt.json")
                if population:
                    try:
                        with open(filename, 'w') as f:
                            json.dump(population[0], f, indent=5)
                    except IOError:
                        pass
            else:
                # For non-final generations, use regular naming
                filename = os.path.join(self.output_path, "results/pops/population_generation_" + str(gen + 1) + ".json")
                try:
                    with open(filename, 'w') as f:
                        json.dump(population, f, indent=5)
                except IOError:
                    pass

                filename = os.path.join(self.output_path, "results/pops_best/population_generation_" + str(gen + 1) + ".json")
                if population:
                    try:
                        with open(filename, 'w') as f:
                            json.dump(population[0], f, indent=5)
                    except IOError:
                        pass

            # Print statistics
            print(f"\n=== Generation {gen + 1} of {self.n_pop} finished. Time Cost: {((time.time()-time_start)/60):.1f} m ===")
            print("Pop Objs: ", end=" ")
            
            if population:
                for i in range(len(population)):
                    obj = population[i]['objective']
                    print(f"{obj:.5f} ", end="")
                # Update stagnation tracking
                worst_objectives.append(population[-1]['objective'])
            else:
                 print("Population is empty.")
                 worst_objectives.append(float('inf'))
                 
            print()


class Methods():
    # Set parent selection method and population management method (R3)
    def __init__(self,paras: Paras, problem: TSPGLS) -> None:
        self.paras = paras      
        self.problem = problem
        
        # Selection methods
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel
        elif paras.selection == 'tournament':
            self.select = tournament
        else:
            print(f"selection method {paras.selection} has not been implemented !")
            exit()

        # Management methods
        if paras.management == "pop_greedy":
            self.manage = pop_greedy
        elif paras.management == 'ls_greedy':
            self.manage = ls_greedy
        elif paras.management == 'ls_sa':
            self.manage = ls_sa
        else:
            print(f"management method {paras.management} has not been implemented !")
            exit()

        
    def get_method(self):
        if self.paras.method == "eoh":   
            return EOH(self.paras,self.problem,self.select,self.manage)
        else:
            print(f"method {self.paras.method} has not been implemented!")
            exit()

class EVOL:
    # Initialization
    def __init__(self, paras: Paras, prob=None, **kwargs):
        print("------------------------------------------------------")
        print("--- Start EoH (TSP) + Automatic Parameter Optimization ---")
        print("------------------------------------------------------")
        create_folders(paras.exp_output_path)
        print("- output folder created -")

        self.paras = paras
        self.prob = prob
        random.seed(2024)

        
    # run methods
    def run(self):
        problemGenerator = Probs(self.paras)
        problem = problemGenerator.get_problem()
        
        if problem is None:
            print("Error: Problem initialization failed.")
            return

        methodGenerator = Methods(self.paras,problem)
        method = methodGenerator.get_method()

        if method:
            method.run()

        print("> End of Evolution! ")
        print("------------------------------------------------------")
        print("---    EoH + AutoParams successfully finished !    ---")
        print("------------------------------------------------------")


# Example Execution Block
if __name__ == "__main__":
    # Parameter initilization #
    paras = Paras() 

    # Set parameters #
    llm_api_endpoint = os.getenv("LLM_API_ENDPOINT")
    llm_api_key = os.getenv("LLM_API_KEY")

    if not llm_api_endpoint or not llm_api_key:
        print("Warning: LLM credentials are not set. Define LLM_API_ENDPOINT and LLM_API_KEY environment variables to enable LLM features.")

    # Set parameters (Using the configuration from the original TSP script)
    paras.set_paras(method = "eoh",
                    problem = "tsp_construct", 
                    
                    # LLM Settings (Ensure these are configured correctly)
                    llm_api_endpoint = llm_api_endpoint,
                    llm_api_key = llm_api_key,
                    llm_model = "gpt-4o-mini",
                    
                    # EC Settings
                    ec_pop_size = 4, 
                    ec_n_pop = 20,
                    
                    # Experiment Settings
                    exp_n_proc = 8,  # multi-core parallel
                    exp_debug_mode = True, # Set to True for detailed logs
                    
                    # Optimization Settings (Integrated)
                    opt_n_trials = 10, # Number of Optuna trials per optimization phase
                    opt_range_factor = 5.0
                   )

    # initilization
    evolution = EVOL(paras)

    # run 
    evolution.run()
    
    print("\nCode integration complete. The execution block is commented out. To run the evolution, uncomment the 'evolution = EVOL(paras)' and 'evolution.run()' lines.")