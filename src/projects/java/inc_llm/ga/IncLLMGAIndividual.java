package inc_llm.ga;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import com.vividsolutions.jts.geom.Point;

import inc_llm.IncLLM;
import nnet.SupervisedUtils;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.som.decay.ConstantDecay;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.LinearDecay;
import spawnn.som.decay.PowerDecay;
import spawnn.utils.DataUtils;
import spawnn.utils.DataUtils.Transform;
import spawnn.utils.SpatialDataFrame;

public class IncLLMGAIndividual extends GAIndividual {


	static int t_max = 1000000;
	static int aMax = 100;
	static int lambda = 1000;
	static double alpha = 0.5;
	static double beta = 0.000005;
	
	static int[] ga = new int[] { 0, 1 };
	static int[] fa = new int[] { 52, 49, 10 };
	static int ta = 7;
	static Dist<double[]> gDist = new EuclideanDist(ga);
	static Sorter<double[]> sorter = new DefaultSorter<double[]>(gDist);
	static List<Entry<List<Integer>, List<Integer>>> cvList;
	static List<double[]> samples;
	static Random r = new Random(0);

	static {
		
		SpatialDataFrame sdf = DataUtils.readSpatialDataFrameFromShapefile(new File("data/election/election2004.shp"), true);
		for (int i = 0; i < sdf.samples.size(); i++) {
			Point p = sdf.geoms.get(i).getCentroid();
			sdf.samples.get(i)[0] = p.getX();
			sdf.samples.get(i)[1] = p.getY();
		}

		samples = sdf.samples;
		DataUtils.transform(samples, fa, Transform.zScore);
		
		cvList = SupervisedUtils.getCVList(10, 1, samples.size());
	}
	
	enum mode {con,lin,pow};
	
	static double[] values = { 0.2, 0.1, 0.05, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001 };

	double[] a = new double[8];
	mode[] m = new mode[4];
	
	public IncLLMGAIndividual() {
		for( int i = 0; i < a.length; i++ )
			a[i] = values[r.nextInt(values.length)];
		for( int i = 0; i < m.length; i++ )
			m[i] = mode.values()[r.nextInt(mode.values().length)];
	}
	
	public IncLLMGAIndividual( double[] a, mode[] m ) {
		this.a = a;
		this.m = m;
	}
	
	@Override
	public GAIndividual mutate() {
		double[] na = Arrays.copyOf(a, a.length);
		mode[] nm = Arrays.copyOf(m, m.length);
		if( r.nextBoolean() ) {
			na[r.nextInt(na.length)] = values[r.nextInt(values.length)];
		} else {
			nm[r.nextInt(nm.length)] = mode.values()[r.nextInt(mode.values().length)];
		}
		return new IncLLMGAIndividual(na, nm);
	}
	
	@Override
	public GAIndividual recombine(GAIndividual mother) {
		IncLLMGAIndividual mo = (IncLLMGAIndividual)mother;
		double[] na = new double[a.length];
		mode[] nm = new mode[m.length];
		for( int i = 0; i < na.length; i++ )
			if( r.nextBoolean() )
				na[i] = a[i];
			else
				na[i] = mo.getA()[i];
		for( int i = 0; i < m.length; i++ )
			if( r.nextBoolean() )
				nm[i] = m[i];
			else
				nm[i] = mo.getMode()[i];		
		return new IncLLMGAIndividual(na, nm);
	}
	
	public double[] getA() {
		return a;
	}
	
	public mode[] getMode() {
		return m;
	}

	double cost = Double.NaN;
	
	@Override
	public double getCost() {
		if( !Double.isNaN(cost) )
			return cost;
				
		DecayFunction[] df = new DecayFunction[4];
		for( int i = 0; i < df.length; i++ ) {
			if( a[i] < a[i+4] ) {
				double tmp = a[i];
				a[i] = a[i+4];
				a[i+4] = tmp;
			}
			if( m[i] == mode.con ) {
				df[i] = new ConstantDecay(a[i]);
			} else if( m[i] == mode.lin ) {
				df[i] = new LinearDecay(a[i],a[i+4]);
			} else {
				df[i] = new PowerDecay(a[i],a[i+4]);
			}
		}
		
		SummaryStatistics ss = new SummaryStatistics();
		for (final Entry<List<Integer>, List<Integer>> cvEntry : cvList) {
			List<double[]> samplesTrain = new ArrayList<double[]>();
			List<double[]> desiredTrain = new ArrayList<double[]>();
			for( int k : cvEntry.getKey() ) {
				samplesTrain.add(samples.get(k));
				desiredTrain.add(new double[]{samples.get(k)[ta]});
			}
			
			List<double[]> samplesVal = new ArrayList<double[]>();
			List<double[]> desiredVal = new ArrayList<double[]>();
			for( int k : cvEntry.getValue() ) {
				samplesVal.add(samples.get(k));
				desiredVal.add(new double[]{samples.get(k)[ta]});
			}
				
			List<double[]> neurons = new ArrayList<double[]>();
			for (int i = 0; i < 2; i++) {
				double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
				neurons.add(Arrays.copyOf(d, d.length));
			}
			
			IncLLM llm = new IncLLM(neurons, 
					df[0],df[1],df[2],df[3],
					sorter, aMax, lambda, alpha, beta, fa, 1, IncLLMGAIndividual.t_max);
					
			for (int t = 0; t < t_max; t++) {
				int idx = r.nextInt(samplesTrain.size());
				llm.train(t, samplesTrain.get(idx), desiredTrain.get(idx));
			}
	
			List<double[]> responseVal = new ArrayList<double[]>();
			for (int i = 0; i < samplesVal.size(); i++)
				responseVal.add(llm.present(samplesVal.get(i)));
	
			ss.addValue(SupervisedUtils.getRMSE(responseVal, desiredVal));
		}
		this.cost = ss.getMean();
		return cost;
	}
	
	public String toString() {
		return Arrays.toString(a)+":::"+Arrays.toString(m);
	}
}
