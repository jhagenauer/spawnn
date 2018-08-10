package hstsom.austria;


import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid2D_Map;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;

public class AuMuGeoSens {
	
	public static void main(String[] args) {
		final int T_MAX = 100000;

		final Random rand = new Random();

		final int runs = 100;
		int threads = 4;

		final List<double[]> samples = DataUtils.readSamplesFromShapeFile(new File("data/sps/munaus_3.shp"), new int[]{}, false);
		
		int[] ta = new int[]{0}; 
		int[] ga = new int[]{1,2}; 
		int[] fa = new int[]{6,8,10,13}; 
		
		final Dist<double[]> eDist = new EuclideanDist();
		final Dist<double[]> fDist = new EuclideanDist( fa);
		final Dist<double[]> gDist = new EuclideanDist( ga);
		final Dist<double[]> tDist = new EuclideanDist( ta);
		
		/*DataUtils.normalizeColumns(samples, ta);
		DataUtils.normalizeGeoColumns(samples, ga);
		DataUtils.normalizeColumns(samples, fa);*/
		
		DataUtils.zScoreColumns(samples, ta);
		DataUtils.zScoreGeoColumns(samples,ga,gDist);
		DataUtils.zScoreColumns(samples, fa);
		
		class Result {
			public int k, dim_x, dim_y;
			public double qe, // quant error
					de, // demo error
					te, // time error
					ge, // geo error
					ti; // time
		}

		ExecutorService es = Executors.newFixedThreadPool(threads);
		List<Future<Result>> futures = new ArrayList<Future<Result>>();

		// evtl auch 18 zu 14
		for (final int DIM_X : new int[]{14} ) {
			for(final int DIM_Y : new int[]{12} ) {
				for (int k = 0; k <= (new Grid2DHex<double[]>(DIM_X,DIM_Y)).getMaxDist(); k++) {
				//for (int k = 0; k < DIM_X; k++) {
					final int K = k;
					for (int i = 0; i < runs; i++) {
	
						futures.add(es.submit(new Callable<Result>() {
							@Override
							public Result call() {
	
								Grid2D_Map<double[]> grid = new Grid2DHex<double[]>(DIM_X, DIM_Y);
								SomUtils.initRandom(grid, samples);						
								
								BmuGetter<double[]> bmuGetter = new KangasBmuGetter<double[]>(gDist, fDist, K);
	
								SOM som = new SOM(new GaussKernel(grid.getMaxDist()), new LinearDecay(0.5, 0.0), grid, bmuGetter);
								long time = System.currentTimeMillis();
								for (int t = 0; t < T_MAX; t++) {
									double[] x = samples.get(rand.nextInt(samples.size()));
									som.train((double) t / T_MAX, x);
								}
								time = System.currentTimeMillis() - time;
	
								Result r = new Result();
								
								r.qe = SomUtils.getMeanQuantError(grid, bmuGetter, eDist, samples);
								r.de = SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples);
								r.te = SomUtils.getMeanQuantError(grid, bmuGetter, tDist, samples);
								r.ge = SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples);
								r.ti = time;
								r.k = K;
								r.dim_x = DIM_X;
								r.dim_y = DIM_Y;
								return r;
							}
						}));
					}
				}
			}
		}
		es.shutdown();

		class ResultContainer {
			public int k, dim_x, dim_y;
			DescriptiveStatistics qeDs = new DescriptiveStatistics();
			DescriptiveStatistics deDs = new DescriptiveStatistics();
			DescriptiveStatistics teDs = new DescriptiveStatistics();
			DescriptiveStatistics geDs = new DescriptiveStatistics();
		}
		
		List<ResultContainer> results = new ArrayList<ResultContainer>();	
		for (Future<Result> f : futures) {
			try {
				Result r = f.get();
				
				boolean found = false;
				for( ResultContainer rc : results ) {
					if( r.k == rc.k && r.dim_x == rc.dim_x && r.dim_y == rc.dim_y ) {
						rc.qeDs.addValue(r.qe);
						rc.deDs.addValue(r.de);
						rc.teDs.addValue(r.te);
						rc.geDs.addValue(r.ge);			
						found = true;
						break;
					}
				}
				
				if( !found ) {
					ResultContainer rc = new ResultContainer();
					rc.k = r.k;
					rc.dim_x = r.dim_x;
					rc.dim_y = r.dim_y;
					rc.qeDs.addValue(r.qe);
					rc.deDs.addValue(r.de);
					rc.teDs.addValue(r.te);
					rc.geDs.addValue(r.ge);	
					results.add(rc);
				}
					
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		
		FileWriter fw = null;
		try {
			fw = new FileWriter("output/aumu_space.csv");
		
			fw.write("radius, x,y, "
					+"minQe,maxQe,meanQe,StdDevQe," 
					+"minDe,maxDe,meanDe,StdDevDe," 
					+"minTe,maxTe,meanTe,StdDevTe," 
					+"minGe,maxGe,meanGe,StdDevGe\n" 
			);
			
			// output
			for( ResultContainer rc : results ) 
				fw.write(rc.k+", "+rc.dim_x+","+rc.dim_y+", "
						+rc.qeDs.getMin()+","+rc.qeDs.getMax()+","+rc.qeDs.getMean()+","+rc.qeDs.getStandardDeviation()+"," 
						+rc.deDs.getMin()+","+rc.deDs.getMax()+","+rc.deDs.getMean()+","+rc.deDs.getStandardDeviation()+","
						+rc.teDs.getMin()+","+rc.teDs.getMax()+","+rc.teDs.getMean()+","+rc.teDs.getStandardDeviation()+","
						+rc.geDs.getMin()+","+rc.geDs.getMax()+","+rc.geDs.getMean()+","+rc.geDs.getStandardDeviation()+"\n"
				);
			
			fw.close();
			
		} catch( Exception e) {
			e.printStackTrace();
		}
	}
}
