import numpy as np
import csv


def getData(file):
	data = []
	with open(file) as csvfile:
		cr = csv.reader(csvfile,delimiter=",")
		for row in cr:
			
			data.append(row)
		return np.asarray(data)



def train(data,iterations):

	a1 = np.random.rand()
	a2 = np.random.rand()
	b = np.random.rand()

	print([a1,a2,b])	
	learningRate = 0.0000000001

	N = float(len(data))

	for e in range(iterations):
		
		for d in data:
			a1_gradient = a2_gradient = b_gradient = 0
			d = list(map(float, d))
			a1_gradient +=  -(2/N) * (d[2] - ((a1 * d[0]) + (a2 * d[1]) + b))*d[0]
			a2_gradient +=  -(2/N) * (d[2] - ((a1 * d[0]) + (a2 * d[1]) + b))*d[1]
			b_gradient +=  -(2/N) * (d[2] - ((a1 * d[0]) + (a2 * d[1]) + b))
		
			a1 = a1 - (learningRate * a1_gradient)
			a2 = a2 - (learningRate * a2_gradient)
			b = b - (learningRate * b_gradient)


	return [a1,a2,b]




def test(params,data):

	error = 0.0
	total_error = 0.0
	N = float(len(data))


	for i in range(0, len(data)):
		x1 = float(data[i,0])
		x2 = float(data[i,1])
		y  = float(data[i,2])
		pred =(params[0]*x1 + params[1]*x2 + params[2])
		error = ((y - pred) ** 2) / N
		total_error += error / N
		print("Error:",error,"Actual:",y,"Predicted:",pred)

	print("Total: ",total_error)

def start():

	data = getData("house.csv")

	traindata = data[:40]
	
	testdata = data[40:]

	params = train(traindata,1000000)

	print(params)
	
	test(params,testdata)



if __name__ == '__main__':

	start()