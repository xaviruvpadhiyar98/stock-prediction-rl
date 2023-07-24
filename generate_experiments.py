from itertools import permutations

def generate_past_hours_permutation():
    items = []
    for item in (permutations(range(-12, 0), 2)):
        if item[0] < item[1]:
            items.append({
                "id": f"start-({item[0]})-end-({item[1]})",
                "value": range(item[0], item[1])
                })
    return items

if __name__=="__main__":
    print(generate_past_hours_permutation())