export PYTHONPATH=$PYTHONPATH:`pwd`
if [ -d ../DATA ]; then
    echo "Data Dir exists"
else 
    mkdir ../DATA
    echo "Data Dir created in ../"
fi 