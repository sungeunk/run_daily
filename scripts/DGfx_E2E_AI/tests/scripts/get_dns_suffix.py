import subprocess


def get_matching_dns_suffix():
    # Run the ipconfig command and capture the output
    result = subprocess.run(['ipconfig', '/all'], capture_output=True, text=True)

    # Split the output into lines
    lines = result.stdout.splitlines()

    # Initialize variables
    dns_suffix_search_list = []
    capture = False

    # Iterate through the lines
    for line in lines:
        if "DNS Suffix Search List" in line:
            capture = True
            # Capture the first line value
            value = line.split(':', 1)[1].strip()
            if value:
                dns_suffix_search_list.append(value)
        elif capture:
            # Check if the line is part of the DNS Suffix Search List
            if line.startswith(' '):
                dns_suffix_search_list.append(line.strip())
            else:
                # Stop capturing if a new section starts
                break

    # Filter to keep only specific strings
    filter_strings = ["amr", "fm", "igk", "iind"]
    for item in dns_suffix_search_list:
        for fs in filter_strings:
            if fs in item:
                return fs

    # Return None if no match is found
    return None

# Call the function and print the result
matching_filter = get_matching_dns_suffix()
if matching_filter:
    print(matching_filter.upper())