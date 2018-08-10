package regioClust;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import spawnn.dist.EuclideanDist;
import spawnn.utils.Clustering;

public class AIC_test {

	public static void main(String[] args) {
		Random r = new Random(0);
		
		List<double[]> samples = new ArrayList<>();
		List<Set<double[]>> cluster = new ArrayList<>();
		int[] fa = new int[]{1};
		int ta = 0;
		
		List<double[]> l0 = new ArrayList<double[]>();
		while( l0.size() < 200 ) {
			double x = r.nextDouble();
			double y = x + r.nextDouble()*0.2-0.1;
			l0.add( new double[]{ y, x } );
		}
		cluster.add( new HashSet<>(l0));
		samples.addAll(l0);
		
		List<double[]> l1 = new ArrayList<double[]>();
		while( l1.size() < 200 ) {
			double x = 1+r.nextDouble();
			double y = x + r.nextDouble()*0.4-0.2;
			l1.add( new double[]{ y, x } );
		}
		cluster.add( new HashSet<>(l1));
		samples.addAll(l1);
				
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("C:/Users/user/git/spawnn/output/chicago.shp"), new int[]{ 1, 2 }, true);
		int[] fa = new int[] { 2,3,7,9,10 };
		int[] ga = new int[]{ 0,1 };
		int ta = 6; // bldUpRt		
		List<double[]> samples = sdf.samples;
		List<Set<double[]>> cluster = new ArrayList<>(Clustering.kMeans(samples, 2, new EuclideanDist(ga)).values());*/
		
		{ // piecewise lm
			LinearModel plm = new LinearModel(samples, cluster, fa, ta, false);
			System.out.println(Arrays.toString(plm.getBeta(0))+","+Arrays.toString(plm.getBeta(1)));
			
			SummaryStatistics ss = new SummaryStatistics();
			for( double d : plm.getResiduals() )
				ss.addValue( d );
			System.out.println("mean: "+ss.getMean()+", var: "+ss.getVariance());
		}
		
		{ // model-matrix lm
			double[] y0 = LinearModel.getY(l0, ta);
			double[][] x0 = LinearModel.getX(l0, fa,true);
			
			double[] y1 = LinearModel.getY(l1, ta);
			double[][] x1 = LinearModel.getX(l1, fa,true);
			
			double[] y = Arrays.copyOf(y0, y0.length+y1.length);
			for( int i = 0; i < y1.length; i++ )
				y[y0.length+i] = y1[i];
			
			double[][] x = new double[x0.length+x1.length][x0[0].length+x1[0].length];
			for( int i = 0; i < x0.length; i++ )
				for( int j = 0; j < x0[0].length; j++ )
					x[i][j] = x0[i][j];
			
			for( int i = 0; i < x1.length; i++ )
				for( int j = 0; j < x1[0].length; j++ )
					x[x0.length+i][x0[0].length+j] = x1[i][j];
						
			DoubleMatrix X = new DoubleMatrix( x );	
			DoubleMatrix Y = new DoubleMatrix( y );
			DoubleMatrix Xt = X.transpose();
			DoubleMatrix XtX = Xt.mmul(X);
			double[] beta = Solve.solve(XtX, Xt.mmul(Y)).data;
			System.out.println( Arrays.toString(beta) ); 
		}
		
		//SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/data/gemeinden_gs2010/gem_dat.shp"), new int[]{ 1, 2 }, true);
		/*SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("R:/Export_Output.shp"), new int[]{ 1, 2 }, true);
		int[] fa = new int[] { 7,  8,  9, 10, 19, 20 };
		int[] ga = new int[]{ 3, 4 };
		int ta = 18; // bldUpRt

		List<Set<double[]>> cluster = new ArrayList<>(Clustering.kMeans(sdf.samples, 1, new EuclideanDist(ga)).values());
		
		{
			long time = System.currentTimeMillis();
			LinearModel plm = new LinearModel(sdf.samples, cluster, "", fa, ta, false);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.getBeta(0)));
			System.out.println(plm.getR2(sdf.samples));
			System.out.println("took: "+(System.currentTimeMillis()-time)/1000.0);
		}
		
		{
			long time = System.currentTimeMillis();
			LinearModel plm = new LinearModel(sdf.samples, cluster, "", fa, ta, true, 0.0);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.getBeta(0)));
			System.out.println( plm.getR2(sdf.samples));
			System.out.println("took: "+(System.currentTimeMillis()-time)/1000.0);
		}
		
		{
			long time = System.currentTimeMillis();
			LinearModel plm = new LinearModel(sdf.samples, cluster, "", fa, ta, true, 0.01);
			System.out.println(plm.numParams+", "+Arrays.toString(plm.getBeta(0)));
			System.out.println( plm.getR2(sdf.samples));
			System.out.println("took: "+(System.currentTimeMillis()-time)/1000.0);
		}*/
	}
}
