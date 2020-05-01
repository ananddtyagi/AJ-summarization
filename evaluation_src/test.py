import rouge_l

from rouge import Rouge

reference = "SAN DIEGO PADRES team notebook"
system = "Williams, who signed a minor league contract, hit .240 with three homers and 12 RBI in 38 games with the Dodgers and Pirates last season."

rouge = Rouge()

def main():
    print(rouge_l.main(reference,system))
    print(rouge.get_scores(system, reference))

main()