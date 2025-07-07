import json
import os
import sys


def is_subsequence(rule, other_args):
    # Ensure both lists are treated as strings for comparison
    rule = list(map(str, rule))
    other_args = list(map(str, other_args))
    
    len_a = len(rule)
    for i in range(len(other_args) - len_a + 1):
        if other_args[i:i+len_a] == rule:
            return True
    return False

def load_validation_rules(platform):
    file_path = f'validation/rules/{platform}.json'
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as file:
        return json.load(file)

def validate_args(platform, workload, backend, model, other_args, rules):

    if workload in rules:
        if backend in rules[workload]:
            if model in rules[workload][backend]:
                if len(rules[workload][backend][model]) == 0: # block entire model
                    return False
                else:
                    # if other_args subsequence is in the rule, block
                    if is_subsequence(rules[workload][backend][model], other_args):
                        return False
                    else:
                        pass
            else:
                pass
        else:
            pass
    else:
        pass

    return True

def main():
    if sys.argv[2] in ("clpeak", "microbench"):
      sys.exit(0)
    
    if len(sys.argv) < 5:
        print("Usage: validate.py <PLATFORM> <WORKLOAD> <BACKEND> <MODEL> <ARGS...>")
        print("Development Error: Refactor to conform to argument structure")
        sys.exit(1)

    platform = sys.argv[1]
    workload = sys.argv[2]
    backend = sys.argv[3]
    model = sys.argv[4]
    other_args = sys.argv[5:]

    rules = load_validation_rules(platform)
    
    if len(rules)==0:
        print(f"No validation rules found for platform: {platform}")
        sys.exit(0)

    if validate_args(platform, workload, backend, model, other_args, rules):
        print(f"Validated arguments, proceeding with test execution")
        sys.exit(0)
    else:
        print(f"Arguments passed are blacklisted for this platform {platform}")
        print(f"For more details of all blacklisted arguments, inspect validation/rules/{platform}.json")
        print(f"To bypass this validation, add 'EXECUTE_MODE=manual' inside .env file")
        sys.exit(10)

if __name__ == "__main__":
    main()
