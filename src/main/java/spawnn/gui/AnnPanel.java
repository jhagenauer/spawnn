package spawnn.gui;

import java.awt.Cursor;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;
import net.miginfocom.swing.MigLayout;
import spawnn.dist.AugmentedDist;
import spawnn.dist.Dist;
import spawnn.dist.EuclideanDist;
import spawnn.dist.WeightedDist;
import spawnn.ng.ContextNG;
import spawnn.ng.NG;
import spawnn.ng.sorter.DefaultSorter;
import spawnn.ng.sorter.KangasSorter;
import spawnn.ng.sorter.Sorter;
import spawnn.ng.sorter.SorterWMC;
import spawnn.ng.utils.NGUtils;
import spawnn.som.bmu.BmuGetter;
import spawnn.som.bmu.BmuGetterContext;
import spawnn.som.bmu.DefaultBmuGetter;
import spawnn.som.bmu.KangasBmuGetter;
import spawnn.som.bmu.SorterBmuGetter;
import spawnn.som.bmu.SorterBmuGetterContext;
import spawnn.som.grid.Grid2D;
import spawnn.som.grid.GridPos;
import spawnn.som.net.ContextSOM;
import spawnn.som.net.SOM;
import spawnn.som.utils.SomUtils;
import spawnn.utils.SpatialDataFrame;

public class AnnPanel extends JPanel implements ChangeListener, ActionListener {
	
	private static final long serialVersionUID = -5917163707063857622L;
	protected JTabbedPane tpANN, tpContextModel;
	protected JTextField trainingCycles, runs;
	protected JPanel somPanel, ngPanel;
	protected JPanel nonePanel, weightedPanel, geoSomPanel, cngPanel, augmentedPanel;
	protected WMCPanel wmcPanel;
	private JButton btnTrain;
		
	private static final int NONE = 0, AUGMENTED = 1, WEIGHTED = 2, GEOSOM = 3, CNG = 4, WMC = 5;
	
	public AnnPanel(Frame parent) {
		super();
		setLayout( new MigLayout("") );
		
		tpANN = new JTabbedPane();
		tpANN.setBorder( BorderFactory.createTitledBorder("Self-Organizing Neural Network") );
		
		ngPanel = new NGPanel();
		tpANN.addTab("NG", ngPanel );

		somPanel = new SOMPanel();
		tpANN.addTab("SOM", somPanel );
		
		tpANN.addChangeListener( this );
				
		// None, Weighted, GeoSOM, CNG, MDMNG
		tpContextModel = new JTabbedPane();
		tpContextModel.setBorder( BorderFactory.createTitledBorder("Spatial Context Model") );
		
		nonePanel = new JPanel();
		tpContextModel.addTab("None", nonePanel);
		
		augmentedPanel = new AugmentedPanel();
		tpContextModel.addTab("Augmented", augmentedPanel);
		
		weightedPanel = new WeightedPanel();
		tpContextModel.addTab("Weighted", weightedPanel );
		
		geoSomPanel = new GeoSOMPanel();
		tpContextModel.addTab("GeoSOM", geoSomPanel );
		
		cngPanel = new CNGPanel();
		tpContextModel.addTab("CNG", cngPanel );
		
		wmcPanel = new WMCPanel();
		tpContextModel.addTab("WMC",wmcPanel );
		
		tpContextModel.addChangeListener(this);
						
		trainingCycles = new JTextField();
		trainingCycles.setText("100000");
		trainingCycles.setColumns(10);
				
		runs = new JTextField();
		runs.setText("1");
		runs.setColumns(2);
						
		btnTrain = new JButton("TRAIN!");
		btnTrain.addActionListener(this);
		
		add( tpANN, "w 50%, span 2, grow, wrap" );
		add(tpContextModel,"span 2, grow, wrap");
		add( new JLabel("Training cycles: "), "split 4" );
		add( trainingCycles, "" );
		add( new JLabel("Runs:"));
		add( runs, "" );
		
		add( btnTrain, "align right" );	
	}

	@Override
	public void stateChanged(ChangeEvent e) {
		if( e.getSource() == tpANN ) // switch NG-SOM happened
			updateContextModelsEnabled();
		//TODO would be nice if train-button is only enabled if distance-matrix is loaded
		/*else if( tpContextModel.getSelectedComponent() == wmcPanel && wmcPanel.getDistanceMap() == null )
			btnTrain.setEnabled(false);
		else 
			btnTrain.setEnabled(true);*/
	}
	
	private void updateContextModelsEnabled() {
		boolean centroidModelsEnabled = gaUsed != null && gaUsed.length > 0;
		tpContextModel.setEnabledAt(AUGMENTED, centroidModelsEnabled);
		tpContextModel.setEnabledAt(WEIGHTED, centroidModelsEnabled);
		tpContextModel.setEnabledAt(CNG,centroidModelsEnabled);	
		
		tpContextModel.setEnabledAt(WMC, dMap != null);
		
		if( tpANN.getSelectedComponent() == somPanel )
			tpContextModel.setEnabledAt(GEOSOM, centroidModelsEnabled);
		else
			tpContextModel.setEnabledAt(GEOSOM, false); // never for ng
		
		if( tpContextModel.getSelectedIndex() == GEOSOM && !tpContextModel.isEnabledAt(GEOSOM) )
			tpContextModel.setSelectedIndex(NONE);
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		if( e.getSource() == btnTrain ) {
						
			int sampleLength = normedSamples.get(0).length;
			Dist<double[]> fDist = new EuclideanDist(fa);
			Dist<double[]> gDist = gaUsed.length > 0 ? new EuclideanDist(gaUsed) : null;
			int t_max = Integer.parseInt(trainingCycles.getText() );
			
			if( tpContextModel.getSelectedComponent() == wmcPanel ) {
				if( dMap == null ) {
					JOptionPane.showMessageDialog(this, "Missing distance matrix! Load or create distance matrix first!", "Missing distance matrix!", JOptionPane.ERROR_MESSAGE);
					return;
				}
			}
			 									
			setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
			for (int run = 0; run < Integer.parseInt(runs.getText()); run++) {

				if (tpANN.getSelectedComponent() == somPanel) { // som
					SOMPanel sp = (SOMPanel) somPanel;

					Grid2D<double[]> grid = ((SOMPanel) somPanel).getGrid();
					SomUtils.initRandom(grid, normedSamples);
					Map<GridPos, Set<double[]>> bmus;

					if (tpContextModel.getSelectedComponent() == wmcPanel) {
						WMCPanel wp = (WMCPanel) wmcPanel;

						Map<GridPos, double[]> initNeurons = new HashMap<GridPos, double[]>();
						for (GridPos p : grid.getPositions()) {
							double[] d = grid.getPrototypeAt(p);
							double[] ns = Arrays.copyOf(d, d.length * 2);
							initNeurons.put(p, ns);
						}
						for (GridPos p : initNeurons.keySet())
							grid.setPrototypeAt(p, initNeurons.get(p));

						List<double[]> prototypes = new ArrayList<double[]>(grid.getPrototypes());
						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : normedSamples)
							bmuHist.put(d, prototypes.get(r.nextInt(prototypes.size())));
						
						SorterWMC s = new SorterWMC(bmuHist, dMap , fDist, wp.getAlpha(), wp.getBeta());
						BmuGetter<double[]> bg = new SorterBmuGetterContext(s);
						ContextSOM som = new ContextSOM(sp.getKernelFunction(), sp.getLearningRate(), grid, (BmuGetterContext) bg, sampleLength);

						s.setHistMutable(true);
						for (int t = 0; t < t_max; t++) {
							double[] x = normedSamples.get(r.nextInt(normedSamples.size()));
							som.train((double) t / t_max, x);
						}
						s.setHistMutable(false);

						bmus = SomUtils.getBmuMapping(normedSamples, grid, bg);
					} else {
						BmuGetter<double[]> bg = null;

						if (tpContextModel.getSelectedComponent() == cngPanel) {
							bg = new SorterBmuGetter<double[]>(new KangasSorter<double[]>(gDist, fDist, ((CNGPanel) cngPanel).getSNS()));
						} else if (tpContextModel.getSelectedComponent() == geoSomPanel) {
							bg = new KangasBmuGetter<double[]>(gDist, fDist, ((GeoSOMPanel) geoSomPanel).getRadius());
						} else if (tpContextModel.getSelectedComponent() == weightedPanel) {
							double w = ((WeightedPanel) weightedPanel).getAlpha();
							Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
							map.put(fDist, 1 - w);
							map.put(gDist, w);
							Dist<double[]> wDist = new WeightedDist<double[]>(map);
							bg = new DefaultBmuGetter<double[]>(wDist);
						} else if (tpContextModel.getSelectedComponent() == augmentedPanel) {
							double a = ((AugmentedPanel) augmentedPanel).getAlpha();
							Dist<double[]> aDist = new AugmentedDist(gaUsed, fa, a);
							bg = new DefaultBmuGetter<double[]>(aDist);
						} else {
							bg = new DefaultBmuGetter<double[]>(fDist);
						}

						SOM som = new SOM(sp.getKernelFunction(), sp.getLearningRate(), grid, bg);
						for (int t = 0; t < t_max; t++) {
							double[] x = normedSamples.get(r.nextInt(normedSamples.size()));
							som.train((double) t / t_max, x);
						}
						bmus = SomUtils.getBmuMapping(normedSamples, grid, bg);
					}
					
					synchronized (bmus) {
						for( TrainingListener tl : _listeners )
							tl.trainingResultsAvailable( new TrainingFinishedSOM(this, normedSamples, bmus, grid,tpContextModel.getSelectedComponent() == wmcPanel));
					}
				} else { // ng
					NGPanel np = (NGPanel) ngPanel;

					NG ng = null;
					Sorter<double[]> s = null;
					if (tpContextModel.getSelectedComponent() == wmcPanel) {
						WMCPanel wp = (WMCPanel) wmcPanel;

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < np.numNeurons(); i++) {
							double[] rs = normedSamples.get(r.nextInt(normedSamples.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2);
							for (int j = rs.length; j < d.length; j++)
								d[j] = r.nextDouble();
							neurons.add(d);
						}

						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : normedSamples)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

						s = new SorterWMC(bmuHist, dMap, fDist, wp.getAlpha(), wp.getBeta());
						ng = new ContextNG(neurons, np.getNeighborhoodRate(), np.getAdaptationRate(), (SorterWMC) s);

					} else {
						if (tpContextModel.getSelectedComponent() == cngPanel) {
							s = new KangasSorter<double[]>(gDist, fDist, ((CNGPanel) cngPanel).getSNS());
						} else if (tpContextModel.getSelectedComponent() == weightedPanel) {
							double w = ((WeightedPanel) weightedPanel).getAlpha();
							Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
							map.put(fDist, 1 - w);
							map.put(gDist, w);
							Dist<double[]> wDist = new WeightedDist<double[]>(map);
							s = new DefaultSorter<double[]>(wDist);
						} else if (tpContextModel.getSelectedComponent() == augmentedPanel) {
							double a = ((AugmentedPanel) augmentedPanel).getAlpha();
							Dist<double[]> aDist = new AugmentedDist(gaUsed, fa, a);
							s = new DefaultSorter<double[]>(aDist);
						} else
							s = new DefaultSorter<double[]>(fDist);

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < np.numNeurons(); i++) {
							double[] rs = normedSamples.get(r.nextInt(normedSamples.size()));
							neurons.add(Arrays.copyOf(rs, rs.length));
						}
							
						ng = new NG(neurons, np.getNeighborhoodRate(), np.getAdaptationRate(), s);
					}

					if (s instanceof SorterWMC)
						((SorterWMC) s).setHistMutable(true);

					for (int t = 0; t < t_max; t++) {
						double[] x = normedSamples.get(r.nextInt(normedSamples.size()));
						ng.train((double) t / t_max, x);
					}

					if (s instanceof SorterWMC)
						((SorterWMC) s).setHistMutable(false);

					Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(normedSamples, ng.getNeurons(), s);

					// build graph, maybe this should better got to NGResultPanel
					Graph<double[], double[]> g = new UndirectedSparseGraph<double[], double[]>();
					for (double[] x : normedSamples) {
						s.sort(x, ng.getNeurons());
						double[] a = ng.getNeurons().get(0);
						double[] b = ng.getNeurons().get(1);

						if (!g.getVertices().contains(a))
							g.addVertex(a);
						if (!g.getVertices().contains(b))
							g.addVertex(b);

						double[] edge = g.findEdge(a, b);
						int count = 1;
						if (edge != null) {
							g.removeEdge(edge);
							count += edge[0];
						}

						if (gDist != null)
							g.addEdge(new double[] { count, fDist.dist(a, b), gDist.dist(a, b) }, a, b);
						else
							g.addEdge(new double[] { count, fDist.dist(a, b) }, a, b);
					}
					
					synchronized (bmus) {
						for( TrainingListener tl : _listeners )
							tl.trainingResultsAvailable( new TrainingFinishedNG(this, normedSamples, bmus, g, tpContextModel.getSelectedComponent() == wmcPanel));
					}
				}
			}
			setCursor(Cursor.getDefaultCursor());
		}
	}
	
	private Random r = new Random();
	private List<double[]> normedSamples;
	private Map<double[], Map<double[], Double>> dMap;
	private int[] fa, gaUsed = null;
	
	public void setTrainingData(List<double[]> normedSamples, SpatialDataFrame spatialData, int[] fa, int[] gaUsed, int[] gaAll, Map<double[], Map<double[], Double>> dMap) {
		this.normedSamples = normedSamples;
		this.fa = fa;
		this.gaUsed = gaUsed;
		this.dMap = dMap;		
		updateContextModelsEnabled();
	}

	private List<TrainingListener> _listeners = new ArrayList<TrainingListener>();
	public synchronized void addTrainingListener(TrainingListener tl) {
		_listeners.add(tl);
	}
}
