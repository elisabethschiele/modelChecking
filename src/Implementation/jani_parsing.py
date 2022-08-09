import json

# replaces all references to globally defined functions with the bodey of those functions within JANI models
# momba unfortunately requires this to be able to parse the model
# was only used on resource gathering - frozen lake did not require this type of adaption

filepath = "../Testing/models/resource-gathering.v2.jani"
functions = {}

with open(filepath, encoding="utf-8") as json_data:
    data = json.load(json_data)

    for i in range (len(data["functions"])):
        functions.update({data["functions"][i]["name"]: data["functions"][i]["body"]})

lines = []
with open("../Testing/models/resource-gathering.v2.jani", "r", encoding="utf-8") as jani_raw:
    for line in jani_raw:
        lines.append(line)

with open("../Testing/models/resource-gathering_parsed.jani", "w", encoding="utf-8") as output:
    i = 0
    while i != len(lines) - 1:
        line = lines[i]
        if lines[i+1].strip().startswith("\"args\":"):
            line_func = lines[i + 2].strip()
            function = line_func.split(" ")[1].split("\"")[1]
            dumped = json.dumps(functions[function], ensure_ascii=False, indent=4)
            if dumped[0] == "{":
                output.write(line)
                function_def = (dumped[1:len(dumped)-1]).encode('utf8').decode()
                output.write(function_def)
                i = i + 3
            else:
                output.write(line.strip()[:len(line.strip())-1])
                output.write(dumped.encode('utf8').decode())
                i = i + 4
                if lines[i+1].strip() != "}":
                    output.write(",")
            output.write("\n")
        else:
            output.write(line)
        i = i+1
    output.write(lines[len(lines)-1])
