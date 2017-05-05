package spawnn.som.net;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.apache.log4j.Logger;

import com.vividsolutions.jts.geom.Point;

import spawnn.UnsupervisedNet;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.decay.LinearDecay;
import spawnn.som.grid.Grid;
import spawnn.som.grid.Grid2DHex;
import spawnn.som.grid.GridPos;
import spawnn.som.kernel.GaussKernel;
import spawnn.som.kernel.KernelFunction;
import spawnn.som.utils.SomUtils;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.GeoUtils;
import spawnn.utils.GeoUtils.GWKernel;
import spawnn.utils.SpatialDataFrame;

public class BatchSOM implements UnsupervisedNet {
	
	private static Logger log = Logger.getLogger(BatchSOM.class);
			
	protected Grid<double[]> grid; 
	protected BmuGetter<double[]> bmuGetter;
	protected KernelFunction nb;
		
	public BatchSOM( KernelFunction nb, Grid<double[]> grid, BmuGetter<double[]> bmuGetter ) {
		this.grid = grid;
		this.bmuGetter = bmuGetter;
		this.nb = nb;
	}
				
	public void train( double t, List<double[]> samples) {
		int vLength = samples.get(0).length;
		
		Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter,false);
		Map<GridPos,double[]> avgMap = new HashMap<GridPos,double[]>();
		for( Entry<GridPos,Set<double[]>> p : bmus.entrySet() )
			avgMap.put(p.getKey(),  DataUtils.getMean( p.getValue() ));
										
		for( GridPos p : grid.getPositions() ) {
						
			double[] a = new double[vLength]; 
			double[] b = new double[vLength];
			for( GridPos gp : bmus.keySet() ) {		
				double f = nb.getValue( grid.dist(p, gp), t) * bmus.get(gp).size();	
				for( int j = 0; j < vLength; j++ ) {
					a[j] += f * avgMap.get(gp)[j];
					b[j] += f;
				}
			}			

			double[] v = new double[vLength];
			for( int j = 0; j < v.length; j++ ) 
				v[j] = a[j]/b[j];		
			grid.setPrototypeAt( p, v );
		}
	}
	
	public void train( double t, List<double[]> samples, Dist<double[]> gDist, GWKernel kernel, double bw ) {
		int vLength = samples.get(0).length;
		Map<GridPos,Set<double[]>> bmus = SomUtils.getBmuMapping(samples, grid, bmuGetter,false);
		
		Map<GridPos,double[]> avgMap = new HashMap<GridPos,double[]>();
		for( Entry<GridPos,Set<double[]>> p : bmus.entrySet() )
			avgMap.put(p.getKey(), GeoUtils.getGWMean( p.getValue(), grid.getPrototypeAt(p.getKey()), gDist, kernel, bw ) );
										
		for( GridPos p : grid.getPositions() ) {
						
			double[] a = new double[vLength]; 
			double[] b = new double[vLength];
			for( GridPos gp : bmus.keySet() ) {		
				double f = nb.getValue( grid.dist(p, gp), t) * bmus.get(gp).size();	
				for( int j = 0; j < vLength; j++ ) {
					a[j] += f * avgMap.get(gp)[j];
					b[j] += f;
				}
			}			

			double[] v = new double[vLength];
			for( int j = 0; j < v.length; j++ ) 
				v[j] = a[j]/b[j];		
			grid.setPrototypeAt( p, v );
		}
	}
	
	@Override
	public void train(double t, double[] x) {
		// TODO call train(t,samples) if batch size reached
	}
	
	public static void main(String[] args) {
		Random r = new Random();
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
			if( i < 10 )
				log.debug(sdf.samples.get(i)[0]+":"+sdf.samples.get(i)[1]);
		}

		int[] ga = new int[] { 0, 1 };
		int[] fa = new int[] { 21, 32, 54, 63, 49 };

		Dist<double[]> fDist = new EuclideanDist(fa);
		Dist<double[]> gDist = new EuclideanDist(ga);

		List<double[]> samples = sdf.samples;
		DataUtils.transform(samples, fa, Transform.zScore);
		
		int T_MAX = 35*samples.size();
		int T_MAX2 = 35;
		
		// Online SOM
		{
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(fDist);

			SOM som = new SOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), new LinearDecay(1.0, 0.0), grid, bmuGetter);
			long time = System.currentTimeMillis();
			for (int t = 0; t < T_MAX; t++) {
				double[] x = samples.get(r.nextInt(samples.size()));
				som.train((double) t / T_MAX, x);
			}
			log.debug("took: "+(System.currentTimeMillis()-time));
			log.debug("Online SOM:");
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
		}
		
		// Batch SOM
		{
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(fDist);

			BatchSOM som = new BatchSOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), grid, bmuGetter);
			long time = System.currentTimeMillis();
			for (int t = 0; t < T_MAX2; t++) 
				som.train((double) t / T_MAX2, samples);
			log.debug("took: "+(System.currentTimeMillis()-time));
			log.debug("Batch SOM:");
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
		}
		
		// Batch GWSom
		for( double bw : new double[]{ 0.001, 0.01, 0.05, 0.1, 0.5, 1.0 } ){
			Grid2DHex<double[]> grid = new Grid2DHex<double[]>(15, 20);
			SomUtils.initRandom(grid, samples);
			BmuGetter<double[]> bmuGetter = new DefaultBmuGetter<double[]>(fDist);

			BatchSOM som = new BatchSOM(new GaussKernel(new LinearDecay(grid.getMaxDist(), 1)), grid, bmuGetter);
			for (int t = 0; t < T_MAX2; t++) 
				som.train((double) t / T_MAX2, samples,gDist,GWKernel.gaussian,bw);
			log.debug("Batch GWSOM "+bw+": ");
			log.debug("fqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, fDist, samples));
			log.debug("sqe: " + SomUtils.getMeanQuantError(grid, bmuGetter, gDist, samples));
			log.debug("te: " + SomUtils.getTopoError(grid, bmuGetter, samples));
		}
	}
}
