


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.Random;

import org.apache.log4j.Logger;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.BatchSOM;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class BasicVsBatch {

	private static Logger log = Logger.getLogger(BasicVsBatch.class);
	
	public static void main(String[] args) {

		Random r = new Random();
		int X_DIM = 20;
		int Y_DIM = 15;
		int T_MAX = 100000;
				
		List<double[]> samples = null;
		try {
			samples = DataUtils.readCSV( new FileInputStream(new File("data/iris.csv") ) );			
			DataUtils.normalize(samples);
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
		
		int EPOCHS = T_MAX/samples.size();
		
		Dist<double[]> eDist = new EuclideanDist();
		
		
		double avgQE = 0;
		double avgTE = 0;
		for( int i = 0; i < 100; i++ ) {
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM );
			SomUtils.initLinear(grid, samples, true);
				
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(eDist);
			
			
			SOM som = new SOM( new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5,0.0), grid, bmuGetter );
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size() ) );
				som.train( (double)t/T_MAX, x );
			}
			
			avgQE += SomUtils.getMeanQuantError( grid, bmuGetter, eDist, samples )/100;
			avgTE += SomUtils.getTopoError( grid, bmuGetter, samples )/100;
		}
		log.debug("Basic SOM: "+avgQE+","+avgTE);
		
		avgQE = 0;
		avgTE = 0;
		for( int i = 0; i < 100; i++ ) {
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(X_DIM, Y_DIM );
			SomUtils.initLinear(grid, samples, true);
				
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(eDist);
			
			
			BatchSOM som = new BatchSOM( new GaussKernel(grid.getMaxDist()), grid, bmuGetter );
			for (int t = 0; t < EPOCHS; t++) 
				som.train( (double)t/EPOCHS, samples );
						
			avgQE += SomUtils.getMeanQuantError( grid, bmuGetter, eDist, samples )/100;
			avgTE += SomUtils.getTopoError( grid, bmuGetter, samples )/100;
		}
		log.debug("Batch SOM: "+avgQE+","+avgTE);
		
	}
}
