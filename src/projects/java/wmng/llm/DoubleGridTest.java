package wmng.llm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.log4j.Logger;

import llm.LLMNG;
import rbf.Meuse;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.som.decay.DecayFunction;
import spawnn.som.decay.PowerDecay;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.utils.GeoUtils;

public class DoubleGridTest {

	private static Logger log = Logger.getLogger(DoubleGridTest.class);
	
	enum model {
		WMNG, NG_LAG, NG, WNG_LAG
	};

	public static void main(String[] args) {		
		long timeAll = System.currentTimeMillis();

		final Random r = new Random();
		
		//int[] nr = new int[]{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48 };
		//int[] nr = new int[]{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 };
		int[] nr = new int[]{ 16 }; 
		
		int maxRun = 4; 
		int threads = 4;
		
		final int[] fa = new int[] { 2 };
		final Dist<double[]> fDist = new EuclideanDist(fa);
		
		String fn = "output/resultDoubleTest_" + maxRun + "_"+nr.length+".csv";
		try {
			Files.write(Paths.get(fn), ("t_max,nrNeurons,nbInit,nbFinal,lr1Init,lr1Final,lr2Init,lr2Final,model,alpha,beta,rmse,r2\n").getBytes());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// load grid data
		final List<GridData> gridData = new ArrayList<GridData>();
		for( int i = 0; i < maxRun; i++ )
			gridData.add( new GridData(
					DoubleGrid2DUtils.createSpDepGrid(50, 50, true), 
					DoubleGrid2DUtils.createSpDepGrid(50, 50, true)
				) );
		
		/*Grid2D<double[]> gridTrain = null;
		for (File fileEntry : new File("/home/julian/publications/geollm/data/grid").listFiles(new FilenameFilter() {
		    public boolean accept(File dir, String name) {
		        return name.toLowerCase().endsWith(".gz");
		    }})) {
			if (fileEntry.isFile()) {
				GZIPInputStream gzis;
				try {
					gzis = new GZIPInputStream(new FileInputStream(fileEntry));
					if( gridTrain == null )
						gridTrain = SomUtils.loadGrid(gzis);
					else {
						gridData.add(new GridData(gridTrain, SomUtils.loadGrid(gzis)));
						gridTrain = null;
					}
					gzis.close();					
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if( gridData.size() == maxRun )
				break;
		}*/
		log.debug("GridData: "+gridData.size());

		for( final int T_MAX : new int[]{ 120000 } )
		for( final int nrNeurons : nr ) // je mehr neuronen, desto größer der unterschied?		
		for( final double nbInit : new double[]{ (double)nrNeurons*2.0/3.0 })
		for( final double nbFinal : new double[]{ 1.0 })
		for( final double lr1Init : new double[]{ 0.6 })
		for( final double lr1Final : new double[]{ 0.01 })
		for( final double lr2Init : new double[]{ 0.6 })
		for (final double lr2Final : new double[] { 0.01 }) {
			List<Object[]> models = new ArrayList<Object[]>();
			//models.add(new Object[] { model.NG, null, null });
			
			models.add(new Object[] { model.NG_LAG, 1, null });
			models.add(new Object[] { model.NG_LAG, 2, null });
			
			/*for (double alpha = 0.0; alpha <= 1.0; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 )
				models.add(new Object[] { model.WNG_LAG, alpha, null });*/
			
			models.add(new Object[] { model.WMNG, 0.8, 0.7 }); // best 16n
			models.add(new Object[] { model.WMNG, 0.8, 0.8 }); // best 16n
			
			for (double alpha = 0.0; alpha <= 1.0; alpha = (double)Math.round( (alpha+0.05) * 100000) / 100000 )
			for (double beta = 0; beta <= 1.0; beta = (double)Math.round( (beta+0.05) * 100000) / 100000 )
				models.add(new Object[] { model.WMNG, alpha, beta });
			
			log.debug("models: "+models.size());
						
			for (final Object[] m : models) {

				long time = System.currentTimeMillis();
				ExecutorService es = Executors.newFixedThreadPool(threads);
				Map<GridData, Future<double[]>> futures = new HashMap<GridData, Future<double[]>>();

				for (final GridData data : gridData) {

					futures.put(data, es.submit(new Callable<double[]>() {

						@Override
						public double[] call() throws Exception {

							List<double[]> samplesTrain = data.samplesTrain;
							List<double[]> desiredTrain = data.desiredTrain;
							
							List<double[]> samplesVal = data.samplesVal;
							List<double[]> desiredVal = data.desiredVal;
							Map<double[], Map<double[], Double>> dMapVal = data.dMapVal;
							
							DecayFunction nbRate = new PowerDecay(nbInit, nbFinal);
							DecayFunction lrRate1 = new PowerDecay(lr1Init, lr1Final);
							DecayFunction lrRate2 = new PowerDecay(lr2Init, lr2Final);

							if (m[0] == model.WMNG) { // WMNG + LLM
								double alpha = (double) m[1];
								double beta = (double) m[2];

								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] rs = samplesTrain.get(r.nextInt(samplesTrain.size()));
									double[] d = Arrays.copyOf(rs, rs.length * 2);
									for (int j = rs.length; j < d.length; j++)
										d[j] = r.nextDouble();
									neurons.add(d);
								}

								Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
								for (double[] d : samplesTrain)
									bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

								Map<double[], Map<double[], Double>> dMapTrain = data.dMapTrain;
								SorterWMC sorter = new SorterWMC(bmuHist, dMapTrain, fDist, alpha, beta);

								ContextNG_LLM ng = new ContextNG_LLM(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);
								ng.useCtx = true;
								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(samplesTrain.size());
									ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
								}

								sorter.setWeightMatrix(dMapVal); // new weight-matrix

								bmuHist.clear();
								for (double[] d : samplesVal)
									bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

								// train histMap
								for (int i = 0; i < 100; i++) {
									List<double[]> rSamplesVal = new ArrayList<double[]>(samplesVal);
									Collections.shuffle(rSamplesVal);
									for (double[] x : rSamplesVal)
										sorter.sort(x, neurons);
								}
								
								List<double[]> responseVal = new ArrayList<double[]>();
								for (double[] x : samplesVal)
									responseVal.add(ng.present(x));
								
								return new double[] { 
										Meuse.getRMSE(responseVal, desiredVal), 
										Meuse.getR2(responseVal, desiredVal),
								};
								
							} else if( m[0] == model.NG_LAG ){
								int lag = (int) m[1];
								List<double[]> lagedSamplesTrain = GeoUtils.getLagedSamples(samplesTrain, data.dMapTrain,lag);
								
								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = lagedSamplesTrain.get(r.nextInt(lagedSamplesTrain.size()));
									neurons.add(Arrays.copyOf(d, d.length));
								}
								
								
								int[] nfa = null;
								if( lag == 1 ) {
									nfa = new int[fa.length*2];
									for( int i = 0; i < fa.length; i++ ) {
										nfa[i] = fa[i];
										nfa[i+fa.length] = fa[i]+samplesTrain.get(0).length;
									}
								} else if( lag == 2 ) {
									nfa = new int[fa.length*3];
									for( int i = 0; i < fa.length; i++ ) {
										nfa[i] = fa[i];
										nfa[i+fa.length] = fa[i]+samplesTrain.get(0).length;
										nfa[i+2*fa.length] = fa[i]+2*samplesTrain.get(0).length;
									}
								} 
								
								Sorter<double[]> sorter = new DefaultSorter<>( new EuclideanDist(nfa));

								LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, nfa, 1);

								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(lagedSamplesTrain.size());
									ng.train((double) t / T_MAX, lagedSamplesTrain.get(j), desiredTrain.get(j));
								}

								List<double[]> lagedSamplesVal = GeoUtils.getLagedSamples(samplesVal, dMapVal,lag);
								List<double[]> responseVal = new ArrayList<double[]>();
								for ( double[] x : lagedSamplesVal )
									responseVal.add(ng.present(x));
								
								return new double[] { 
										Meuse.getRMSE(responseVal, desiredVal), 
										Meuse.getR2(responseVal, desiredVal),
								};
							} else if( m[0] == model.WNG_LAG ){
								double alpha = (double)m[1];
								
								List<double[]> lagedSamplesTrain = GeoUtils.getLagedSamples(samplesTrain, data.dMapTrain);
								
								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = lagedSamplesTrain.get(r.nextInt(lagedSamplesTrain.size()));
									neurons.add(Arrays.copyOf(d, d.length));
								}
								
								int[] faLag = new int[fa.length];
								for( int i = 0; i < fa.length; i++ )
									faLag[i] = samplesTrain.get(0).length + fa[i];
								
								int[] nfa = new int[fa.length*2];
								for( int i = 0; i < fa.length; i++ ) {
									nfa[i] = fa[i];
									nfa[fa.length + i] = faLag[i];
								}
								
								Map<Dist<double[]>,Double> m = new HashMap<Dist<double[]>,Double>();
								m.put(fDist, alpha);
								m.put(new EuclideanDist(faLag), 1.0-alpha);
								
								Sorter<double[]> sorter = new DefaultSorter<>( new WeightedDist<>(m));

								LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, nfa, 1);

								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(lagedSamplesTrain.size());
									ng.train((double) t / T_MAX, lagedSamplesTrain.get(j), desiredTrain.get(j));
								}

								List<double[]> lagedSamplesVal = GeoUtils.getLagedSamples(samplesVal, dMapVal);
								List<double[]> responseVal = new ArrayList<double[]>();
								for ( double[] x : lagedSamplesVal )
									responseVal.add(ng.present(x));
								
								return new double[] { 
										Meuse.getRMSE(responseVal, desiredVal), 
										Meuse.getR2(responseVal, desiredVal),
								};
							} else { // NG
								List<double[]> neurons = new ArrayList<double[]>();
								for (int i = 0; i < nrNeurons; i++) {
									double[] d = samplesTrain.get(r.nextInt(samplesTrain.size()));
									neurons.add(Arrays.copyOf(d, d.length));
								}
														
								Sorter<double[]> sorter = new DefaultSorter<>( new EuclideanDist(fa));
								LLMNG ng = new LLMNG(neurons, nbRate, lrRate1, nbRate, lrRate2, sorter, fa, 1);

								for (int t = 0; t < T_MAX; t++) {
									int j = r.nextInt(samplesTrain.size());
									ng.train((double) t / T_MAX, samplesTrain.get(j), desiredTrain.get(j));
								}

								List<double[]> responseVal = new ArrayList<double[]>();
								for ( double[] x : samplesVal )
									responseVal.add(ng.present(x));
								
								return new double[] { 
										Meuse.getRMSE(responseVal, desiredVal), 
										Meuse.getR2(responseVal, desiredVal),
								};
							}
						}
					}));
				}
				es.shutdown();

				// get statistics
				try {
					DescriptiveStatistics ds[] = null;
					for (Entry<GridData, Future<double[]>> ff : futures.entrySet()) {
						
						double[] ee = ff.getValue().get();
						if (ds == null) {
							ds = new DescriptiveStatistics[ee.length];
							for (int i = 0; i < ee.length; i++)
								ds[i] = new DescriptiveStatistics();
						}
						for (int i = 0; i < ee.length; i++)
							ds[i].addValue(ee[i]);
					}

					// write statistics
					String s = T_MAX + "," + nrNeurons + "," + nbInit + "," + nbFinal + "," + lr1Init + "," + lr1Final + "," + lr2Init + "," + lr2Final + "," + Arrays.toString(m).replaceAll("\\[", "").replaceAll("\\]", "");
					for (int i = 0; i < ds.length; i++)
						s += "," + ds[i].getMean();
					s += "\n";
					System.out.print(s);
					s = s.replace("null", "0.0");
					Files.write(Paths.get(fn), s.getBytes(), StandardOpenOption.APPEND);
				} catch (IOException e) {
					e.printStackTrace();
				} catch (InterruptedException e) {
					e.printStackTrace();
				} catch (ExecutionException e) {
					e.printStackTrace();
				}
				log.debug("model took: " + (System.currentTimeMillis() - time) / 1000.0 + " sec");
			}
		}
		log.debug("took: " + (System.currentTimeMillis() - timeAll) / 1000.0 / 60.0 + " min");
	}

	public static double[] getStripped(double[] d, int[] fa) {
		double[] nd = new double[fa.length];
		for (int i = 0; i < fa.length; i++)
			nd[i] = d[fa[i]];
		return nd;
	}
	
	public static Grid2D<double[]> getSurrounding( GridPos c, Grid2D<double[]> grid, int dist, boolean center ) {
		Set<GridPos> done = new HashSet<GridPos>();
		List<GridPos> openList = new ArrayList<GridPos>();
		openList.add(c);
		while( !openList.isEmpty() ) {
			GridPos cur = openList.remove(0);
			done.add(cur);
						
			for( GridPos nb : grid.getNeighbours(cur) ) 
				if( !done.contains(nb) && grid.dist(c, nb) <= dist )
					openList.add(nb);	
		}
		
		int xSize = grid.getSizeOfDim(0);
		int ySize = grid.getSizeOfDim(1);
		
		Grid2D<double[]> g = new Grid2D<double[]>(0,0);
		for( GridPos p : done ) {
			if( grid.dist(p, c) != dist )
				continue;
			if( center ) {
				int x = p.getPos(0)- c.getPos(0);
				int y = p.getPos(1) - c.getPos(1);
		
				if (x > dist)
					x -= xSize;
				if (y > dist)
					y -= ySize;
				
				if (x < -dist)
					x += xSize;
				if (y < -dist)
					y += ySize;
				
				g.setPrototypeAt(new GridPos(x, y),grid.getPrototypeAt(p));
			} else
				g.setPrototypeAt(p,grid.getPrototypeAt(p));
		}
		return g;
	}
			
	public static <T> double getDistError(Map<T, Set<double[]>> bmus, Grid2D<double[]> grid, int dist, int fa ) {
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for( Entry<T,Set<double[]>> e : bmus.entrySet() ) {
			
			if( e.getValue().isEmpty() )
				continue;
			
			// get receptive fields
			Set<Grid2D<double[]>> rfs = new HashSet<Grid2D<double[]>>();
			for( double[] d : e.getValue() )
				rfs.add( getSurrounding( grid.getPositionOf(d), grid, dist, true));
			
			// mean receptive fields
			Grid2D<double[]> mrf = new Grid2D<double[]>(0,0);
			for( GridPos p : rfs.iterator().next().getPositions() ) {
				double[] m = new double[rfs.iterator().next().getPrototypeAt(p).length];
				for( Grid2D<double[]> g : rfs ) {
					double[] pt = g.getPrototypeAt(p);
					for( int i = 0; i < pt.length; i++ )
						m[i] += pt[i]/rfs.size();
				}
				mrf.setPrototypeAt(p, m);
			}
			
			// get mean diff
			for( Grid2D<double[]> rf : rfs ) 
				for( GridPos p : rf.getPositions() )
					ds.addValue( Math.pow( rf.getPrototypeAt(p)[fa] - mrf.getPrototypeAt(p)[fa],2 ) );			
		}	
		return Math.sqrt(ds.getMean());
	}
}
