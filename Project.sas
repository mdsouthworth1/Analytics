/*Matthew Southworth
Project Assignment */
proc import datafile="/home/u59384788/sasuser.v94/cars.csv" dbms=csv out=cars 
		replace;
RUN;

data cars;
	set cars;
	ln_price=log(price_USD);
	ln_duration=log(duration_listed);
run;

proc sort data=cars;
	by body_type odometer_value;
run;

proc print data=cars (obs=10);
run;

proc means data=cars skewness kurtosis;
run;

Proc Boxplot data=Cars;
	plot price_USD*body_type /outbox=car_data boxstyle=schematic;
run;

proc Gchart data=cars;
	/* general bar charting proc*/
	pie manufacturer_name 
		/type=pct /*Type=freq gives counts, pct gives percentages*/
		slice=arrow;

	/*The slice command designates how the slices are labeled,
	and the arrow is an option that makes the label point to its slice */
	run;

proc Gchart data=cars;
	/* general bar charting proc*/
	pie body_type /type=pct /*Type=freq gives counts, pct gives percentages*/
	slice=arrow;

	/*The slice command designates how the slices are labeled,
	and the arrow is an option that makes the label point to its slice */
	run;

proc univariate data=cars;
	histogram price_USD ln_price odometer_value engine_capacity number_of_photos;
	qqplot price_USD;
run;

proc reg data=cars;
	model price_USD=odometer_value year_produced engine_capacity 
		number_of_photos/vif;
	run;

proc reg data=cars;
	model ln_price=odometer_value year_produced engine_capacity number_of_photos 
		ln_duration/vif lackfit;
	run;

PROC glm DATA=cars;
	class transmission color drivetrain body_type;
	MODEL ln_price=odometer_value year_produced engine_capacity transmission color 
		drivetrain body_type number_of_photos/ ss3;
	run;

proc anova data=cars;
	class transmission color drivetrain body_type engine_type engine_fuel;
	model ln_price=transmission color drivetrain body_type engine_type engine_fuel;
	means transmission color drivetrain body_type engine_type engine_fuel / tukey 
		lines cldiff;
	run;

proc factor data=cars corr scree ev method=principal;
	var odometer_value year_produced engine_capacity number_of_photos 
		duration_listed;
run;