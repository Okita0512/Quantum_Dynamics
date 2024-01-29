mkdir summary

# for i in 600 800 900 1000 1050 1100 1120 1140 1150 1160 1170 1180 1200 1250 1300 1400 1500 1600 1700 1800
for i in 1170
do

cd $i

cp prop-rho.dat ../summary/$i.dat

cd ../

done
