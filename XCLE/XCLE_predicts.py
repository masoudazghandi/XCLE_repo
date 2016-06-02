import flask
from flask import Response
import numpy as np
import os, pickle
from sklearn.externals import joblib
import pandas as pd
import time
import sqlalchemy
from uuid import uuid1
from datetime import datetime
import sys
import logging
from logging import FileHandler
from logging import Formatter

DB_TALK = True
DEBUG_FLAG = False


if '--nodb' in sys.argv:
		DB_TALK = False

if '--debug' in sys.argv:
		DEBUG_FLAG = True


app = flask.Flask(__name__)

# Configure Logger Handler for logging errors
file_handler = FileHandler('XCLE/log/error_log.txt',mode = 'w+', encoding = 'utf8')
file_handler.setFormatter(Formatter(
'%(asctime)s %(levelname)s: %(message)s '
'[in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.ERROR)
app.logger.addHandler(file_handler)




model_dir = 'XCLE/static/Models/'
risk_f = pickle.load(open(model_dir + 'risk_function.pkl', 'rb'))

all_files = os.listdir(model_dir)
xr_files = [file for file in all_files if file[:2] =='xr'] 
load_files = [file for file in xr_files if file[-3:] =='pkl']

names = np.array([file[2:-4] for file in load_files if "_feats" not in file])



model_names = np.array(
		[name for name in names if 'baseline' not in name])
model_files = np.array(
		['xr' + model_name +'.pkl' for model_name in model_names])
feature_files = np.array(
		[file for file in all_files if file[-9:] =='feats.pkl'])

baseline_model_names = np.array(
		[name for name in names if 'baseline' in name])
baseline_model_files =  np.array(
		['xr'+b_name+'.pkl' for b_name in baseline_model_names])
baseline_model_names = np.array(
		[name.replace('_baseline','') for name in baseline_model_names])

model_scores = pickle.load(open(model_dir + 'model_acc_scores.pkl', 'rb'))
adj_model_scores = {key:(value-0.5)/.5 for key,value in model_scores.items()}

PREDICTORS = {}
for name, model,features in zip(model_names, model_files, feature_files):
	PREDICTORS[name] = {'features': joblib.load(model_dir + features),
		'model': joblib.load(model_dir + model)}

BASE_PREDICTORS = {}
for base_name, base_file in zip(baseline_model_names, baseline_model_files):
	BASE_PREDICTORS[base_name] = {'model': pickle.load(
		open(model_dir + base_file, 'rb'))}

if DB_TALK:
		DB_ENGINE = sqlalchemy.create_engine("mssql+pyodbc://CTCERTXCLEPROD32")

ALLOWED_EXTENSIONS = ['xls','xlsx']

# Read the top pubs used as features for Bug Classification
pubs = sorted(pickle.load(open(model_dir + 'pub_features.pkl','rb')))

fuzzy_months = [
	['fuzzy_Dec', 'fuzzy_Jan', 'fuzzy_Feb'], 
	['fuzzy_Jan', 'fuzzy_Feb', 'fuzzy_Mar'],
	['fuzzy_Feb', 'fuzzy_Mar', 'fuzzy_Apr'],
	['fuzzy_Mar', 'fuzzy_Apr', 'fuzzy_May'],
	['fuzzy_Apr', 'fuzzy_May','fuzzy_Jun'],
	[ 'fuzzy_May','fuzzy_Jun','fuzzy_Jul'],
	['fuzzy_Jun','fuzzy_Jul','fuzzy_Aug'],
	['fuzzy_Jul','fuzzy_Aug', 'fuzzy_Sep'],
	['fuzzy_Aug', 'fuzzy_Sep', 'fuzzy_Oct'],
	['fuzzy_Sep', 'fuzzy_Oct','fuzzy_Nov'],
	['fuzzy_Oct','fuzzy_Nov', 'fuzzy_Dec'], 
	['fuzzy_Nov', 'fuzzy_Dec', 'fuzzy_Jan']]
				
if DB_TALK:
	db_submit_types = {
		"Raw_Test_Order": sqlalchemy.dialects.mssql.base.INTEGER,
		"Test_Case": sqlalchemy.dialects.mssql.base.VARCHAR,
		"Fail_Probability": sqlalchemy.dialects.mssql.base.DECIMAL, 
		"Accuracy": sqlalchemy.dialects.mssql.base.DECIMAL,
		"Order_Metric": sqlalchemy.dialects.mssql.base.DECIMAL,
		"Actual": sqlalchemy.dialects.mssql.base.BIT,
		"Kickoff_Date": sqlalchemy.dialects.mssql.base.DATETIME,
		"XCID": sqlalchemy.dialects.mssql.base.VARCHAR,
		"Prediction_Batch_Id": 
				sqlalchemy.dialects.mssql.base.UNIQUEIDENTIFIER,
		"Created_By": sqlalchemy.dialects.mssql.base.NVARCHAR,
		"Created_Date": sqlalchemy.dialects.mssql.base.DATE,
		"Test_Recommended": sqlalchemy.dialects.mssql.base.BIT}

	log_submit_types = {
		"SubmissionType": sqlalchemy.dialects.mssql.base.VARCHAR,
		"Publisher": sqlalchemy.dialects.mssql.base.NVARCHAR, 
		"XCID": sqlalchemy.dialects.mssql.base.VARCHAR,
		"MultiplayerGame": sqlalchemy.dialects.mssql.base.BIT,
		"BlockbusterGame": sqlalchemy.dialects.mssql.base.BIT,
		"Month": sqlalchemy.dialects.mssql.base.INTEGER,
		"KickOffDate": 
				sqlalchemy.dialects.mssql.base.DATE,
		"IndependentGame": sqlalchemy.dialects.mssql.base.BIT,
		"Prediction_Batch_Id": 
				sqlalchemy.dialects.mssql.base.UNIQUEIDENTIFIER,
		"TPG_Processed": sqlalchemy.dialects.mssql.base.BIT}

db_col_name_replace = {
	"Raw Test Order":"Raw_Test_Order", 
	"Test Case":"Test_Case", 
	"Fail Probability": "Fail_Probability", 
	"Order Metric":"Order_Metric",
	"Test Recommended": "Test_Recommended", 
	"Date":"Kickoff_Date"}

log_db_col_name_replace = {
	"top_pubs":"Publisher", 
	"Mltplyr": "MultiplayerGame",
	"Blckbster":"BlockbusterGame",
	"Ind": "IndependentGame", 
	"sub_type":"SubmissionType",
	"kickoff_month": "Month",
	"user_id": "Created_By"
	}

# Set first unique batch id
batch_id = str(uuid1())

# Get current date 
date_str = time.strftime("%m-%d-%Y")
output_filename = 'XCLE/dynamic_results/Results_' + date_str

@app.route("/")
def viz_page():
		""" Homepage: serve our form page, form.html
		"""
		with open("XCLE/templates/form.html", 'r') as viz_file:
				return viz_file.read()

def calculate_risk(order_metric):
		if np.isnan(order_metric):
			return(np.nan)
		else:
			risk = risk_f([order_metric])[0]
			if risk > 100:
				return(100.0000)
			else:
				return(risk)

def feature_map(form_features, model_feats):

		# map input feature month to fuzzy months 
		num_subs = len(form_features['sub_type'])
		test = pd.DataFrame(0, columns = model_feats, index =range(num_subs))
		for idx, month in  enumerate(form_features['kickoff_month']):
			fms = fuzzy_months[int(month)]
			for fm in fms:
				if fm in test.columns:
						test.loc[idx, fm] = 1
		for idx, pub in enumerate(form_features['top_pubs']):
			if pub in test.columns:
					test.loc[idx, pub] = 1 
						
		for idx, (st, sn) in enumerate(zip(form_features['sub_type'],
			form_features['Pass_Num'])):
			str1 = 'pass_#'
			if int(sn) > 5:
					str2 = ' >5'
			else:
					str2 = ' =' + sn
			str12 = str1+str2
			if str12 in test.columns:
					test.loc[idx, str12] = 1

		commons = list(set(test.columns).intersection(form_features.keys()))
		for common in commons:
			test[common] = [int(f) for f  in form_features[common]]
		return test .values

@app.route("/info", methods=["POST"])
def info():
		"""  When A POST request with text is made, respond with the info requested
		"""
		info_response = {}

		# Get info for dynamic data entry 
		info_req = flask.request.json
		if 'pubs' in info_req.keys():
			info_response.update({'pubs' : pubs})
		if 'date' in info_req.keys():
			info_response.update({'date' : date_str})
		return flask.jsonify(info_response)

# Generate reports in html and excel and save 
@app.route("/score", methods=["POST"])
def score():

		"""  When A POST request with json data is made to this uri,
				 Read the example from the json, predict probability and
				 send it with a response
		"""
		global batch_id
		global log_df

		try:
		# Get decision score for the submission request
			
			form_features = flask.request.json

			form_features = pd.DataFrame(form_features).drop_duplicates().to_dict('list')
			user_ids = form_features['user_id']

			# report_type = form_features.pop('report_type')
			XCID_strs = form_features.pop('xcid_1')
			dates = [XCID_str.split('-')[1] for XCID_str in XCID_strs]



			years = [int(date[0:4]) for date in dates]
			months = [int(date[4:6]) for date in dates]
			days = [int(date[6:]) for date in dates]
			kickoff_dates = [datetime(year, month, day).date() for year, month, day in zip(years,months,days)]
			form_features['kickoff_month'] = months

			probs = {}
			for name, trained in PREDICTORS.items():
				model_feats = trained['features']
				clf = trained['model']
				test = feature_map(form_features, model_feats)
				probs[name] = clf.predict_proba(test)[:,1]


			# order and format report
			adj_probs = {key:(value-0.5)/.5 for key,value in probs.items()}  

			batch_results = {}

			for xcid_idx, XCID_str in enumerate(XCID_strs):

					output_df = pd.DataFrame(index = adj_probs.keys(),
							columns = ['Fail Probability', 'Accuracy', 'Order Metric'])

					for xr in output_df.index:
						if (xr in probs.keys()) and (xr in model_scores.index):
							if model_scores.loc[xr,'Sample_Size'] > 2:
								
								output_df.loc[xr, 'Fail Probability'] = probs[xr][xcid_idx]
								output_df.loc[xr, 'Accuracy'] = model_scores.loc[xr,'Accuracy']
								output_df.loc[xr, 'Order Metric'] = (
										adj_probs[xr][xcid_idx] * output_df.loc[xr, 'Accuracy'])
							else:

								output_df.loc[
										xr, 'Fail Probability'] = np.nan
								output_df.loc[xr, 'Accuracy'] = np.nan
								output_df.loc[xr, 'Order Metric'] = np.nan

					base_probs = {}
					for name, trained in BASE_PREDICTORS.items():
						base_probs[name] = trained['model']['y_prob']

					# order and format report - Baseline
					adj_b_probs = {key:(value-0.5)/.5 for key,value in base_probs.items()}  
					base_output_df = pd.DataFrame(index = adj_b_probs.keys(),
						columns = ['Fail Probability', 'Accuracy', 'Order Metric'])

					for title in base_output_df.index:
						xr = title.split('_')[0]
						if (title in base_probs.keys()) and (xr in model_scores.index):
							if model_scores.loc[xr,'Sample_Size'] > 2:
									base_output_df.loc[
											title, 'Fail Probability'] = base_probs[title]
									base_output_df.loc[title, 'Accuracy'] = (
											model_scores.loc[xr,'Accuracy'])
									base_output_df.loc[title, 'Order Metric'] = (
													adj_b_probs[title] *
													 base_output_df.loc[title, 'Accuracy'])
							else:
								base_output_df.loc[
										title, 'Fail Probability'] = np.nan
								base_output_df.loc[title, 'Accuracy'] = np.nan
								base_output_df.loc[title, 'Order Metric'] = np.nan


					output_df = pd.concat([output_df, base_output_df])
					output_df['Risk'] = output_df['Order Metric'].apply(calculate_risk)

					output_df['Test Recommended'] = output_df['Risk'] > 40
					low_samp_bidx = output_df['Order Metric'].isnull()
					output_df.loc[low_samp_bidx,'Test Recommended'] = True
					output_df.loc[low_samp_bidx, 'Fail Probability'] = np.nan
					output_df.loc[low_samp_bidx, 'Accuracy'] = np.nan
					output_df.loc[low_samp_bidx, 'Order Metric'] = np.nan
					output_df['Low_Sample_Size'] = 0
					output_df.loc[low_samp_bidx, 'Low_Sample_Size'] = 1
					output_df['Date'] = kickoff_dates[xcid_idx]
					output_df['XCID'] = XCID_str
					output_df['Created_By'] = user_ids[xcid_idx]
					output_df.sort_values(
						by = 'Order Metric', ascending = False, inplace = True,
						na_position = 'first')
					output_df.reset_index(inplace = True)
					output_df.rename(columns = {'index':'Test Case'}, inplace = True)
					output_df.set_index(output_df.index + 1, inplace = True)
					first_idx = output_df['Fail Probability'].isnull()
					ser_index = output_df.index.to_series()-sum(first_idx)
					ser_index.loc[first_idx] = 0
					output_df.set_index(ser_index.values, inplace=True)
					output_df.index.name = 'Raw Test Order'
					output_df = output_df.loc[:,['XCID','Test Case', 'Fail Probability', 'Accuracy', 'Order Metric',
						'Test Recommended', 'Low_Sample_Size',  'Date', 'Created_By', 'Risk']]
					batch_results[XCID_str] = output_df.copy()


			batch_df = pd.concat(list(batch_results.values()))

			####### For pre-TPG reporting #############
			#export to excel
			# with pd.ExcelWriter(output_filename + '.xlsx') as writer:
			# 	for xcid_idx, xcid in enumerate(batch_results.keys()):
			# 		batch_results[xcid].to_excel(
			# 				writer, sheet_name = xcid, float_format = "%.3f")

			# #export to html
			# with open('XCLE/static/Results.html', 'w') as fp:

			# 		batch_df.to_html(fp, float_format = lambda x: '%.3f' % x)
			####### For pre-TPG reporting #############

			pickle.dump(batch_df,  open('./XCLE/static/Results.pkl', 'wb'))
			log_df =  pd.DataFrame(form_features, index=[XCID_strs])
			log_df.index.name = 'XCID'
			log_df.rename(columns = log_db_col_name_replace, inplace=True)

			# Change global batch id for each prediction set 
			batch_id = str(uuid1())
			log_df['Prediction_Batch_Id'] = batch_id
			log_df['KickOffDate'] = kickoff_dates

			pickle.dump(log_df,  open('XCLE/static/Parameters.pkl', 'wb'))
			results = {'result': 'success' }
		except:
			results = {'result': 'failed' }

		return flask.jsonify(results)

@app.route("/send_to_db", methods=["POST"])
def res_to_db():

		"""  When A POST request with json data is made to this uri,
				 read the prediction from pickled data, and send to db
		"""
		try:
			batch_df = pickle.load(open('XCLE/static/Results.pkl', 'rb'))
			log_df = pickle.load(open('XCLE/static/Parameters.pkl', 'rb'))

			batch_df["Fail Probability"] = batch_df["Fail Probability"].replace(
				{'low samples': np.nan}).astype('float')
			batch_df["Accuracy"] = batch_df["Accuracy"].replace(
				{'low samples': np.nan}).astype('float')
			batch_df["Order Metric"] = batch_df["Order Metric"].replace(
				{'low samples': np.nan}).astype('float')
			batch_df["Test Recommended"] = batch_df["Test Recommended"].astype(
				'int')

			batch_df.rename(columns = db_col_name_replace, inplace=True)

			batch_df["Prediction_Batch_Id"] = batch_id
			batch_df["Actual"] = np.nan
			batch_df.index.name = "Raw_Test_Order"   

			if DB_TALK:        
				log_df.to_sql("PredictionParameterLog", DB_ENGINE, if_exists = 'append',
					dtype = log_submit_types)       
				batch_df.to_sql("PredictionData", DB_ENGINE,
					if_exists = 'append', dtype = db_submit_types)

			results = {'result': 'success' }			

		except:
			results = {'result': 'failed' }

		return flask.jsonify(results)


@app.route("/download_doc")
def download_doc():
	excelDownload = open('XCLE/static/XCLE_documentation.pdf','rb').read()
	return Response(
		excelDownload,
		mimetype="application/pdf",
		headers={"Content-disposition":
			"attachment; filename=" + "XCLE_documentation.pdf"})

if __name__ == "__main__":

	#--------- RUN WEB APP SERVER ------------#

	# Start the app server on port 5000 in debug mode
	if DEBUG_FLAG:
			# localhost  instance, with debugging options turned on
			app.debug = True
			app.run( )
	else:

		app.debug = False
		app.run(host="0.0.0.0", port=88)
