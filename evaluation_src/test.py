import rouge_l

from rouge import Rouge

# reference = "SAN DIEGO PADRES team notebook"
# system = "Williams, who signed a minor league contract, hit .240 with three homers and 12 RBI in 38 games with the Dodgers and Pirates last season."

reference = "blah asfe this is cool"
system = "."

rouge = Rouge()

def main():

    try:
        print(rouge.get_scores(system, reference))
    except ValueError:
        h = 0

main()