#!/bin/bash
START=1
END=30

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_110_final_weak 110 "$c" "weak"
done

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_126_final_weak 126 "$c" "weak"
done

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_2_final_weak 2 "$c" "weak"
done

for (( c=$START; c<=$END; c++ ))
do
    python automa_pykoopman.py exp_rule_1_final_weak 1 "$c" "weak"
done