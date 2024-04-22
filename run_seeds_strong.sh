#!/bin/bash
START=1
END=30

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_110_final 110 "$c" "strong"
done

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_126_final 126 "$c" "strong"
done

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_2_final 2 "$c" "strong"
done

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_1_final 1 "$c" "strong"
done