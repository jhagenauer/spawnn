package spawnn.gui;

import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JTabbedPane;

import net.miginfocom.swing.MigLayout;

import org.apache.log4j.Logger;

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
import spawnn.utils.DataUtils;
import spawnn.utils.GeoUtils;
import spawnn.utils.SpatialDataFrame;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.graph.UndirectedSparseGraph;

public class SpawnnGui extends JFrame implements PropertyChangeListener, ActionListener {

	private static Logger log = Logger.getLogger(SpawnnGui.class);
	private static final long serialVersionUID = -2728973134956990131L;

	private DataPanel dataPanel;
	private AnnPanel annPanel;
	private Random r = new Random();

	private JTabbedPane tp;
	private JMenuItem aboutItem, quitItem;

	SpawnnGui() {
		super("Spawnn");

		// Menu Bar
		JMenuBar mb = new JMenuBar();
		mb.setLayout(new MigLayout("insets 0"));
		setJMenuBar(mb);

		JMenu fileMenu = new JMenu("File");
		
		quitItem = new JMenuItem("Quit");
		quitItem.addActionListener(this);
		fileMenu.add(quitItem);

		JMenu helpMenu = new JMenu("Help");
		aboutItem = new JMenuItem("About");
		aboutItem.addActionListener(this);
		helpMenu.add(aboutItem);

		mb.add(fileMenu, "push");
		mb.add(helpMenu);

		// Tabs
		tp = new JTabbedPane();
		dataPanel = new DataPanel(this);
		dataPanel.addPropertyChangeListener(this);

		tp.addTab("Data", dataPanel);

		annPanel = new AnnPanel();
		annPanel.setEnabledCentroidModels(false);
		annPanel.addPropertyChangeListener(this);

		tp.addTab("ANN", annPanel);
		tp.setEnabledAt(1, false);

		add(tp);

		pack();
		setSize(1245, 700);
		setMinimumSize(new Dimension(1245, 700)); // ugly, but needed because of ng results minimum size (vv)

		setVisible(true);
	}

	public static void main(String[] args) {
		/*try {
			UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
		} catch (Exception e) {
			try {
				UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
			} catch (Exception ex) {
				ex.printStackTrace();
			}
		}*/
		new SpawnnGui();
	}

	int numSom = 0;
	int numNg = 0;

	@Override
	public void propertyChange(PropertyChangeEvent e) {
		if( e.getPropertyName().equals(AnnPanel.APPLY_EXISTING) && (Boolean)e.getNewValue() ) {
			int[] fa = dataPanel.getFA();
			int[] ga = dataPanel.getGA(false);

			List<double[]> samples = dataPanel.getNormedSamples();
			SpatialDataFrame origData = dataPanel.getSpatialData();
			
			if (dataPanel.getSpatialData() == null) {
				JOptionPane.showMessageDialog(this, "No data loaded yet!", "No data!", JOptionPane.ERROR_MESSAGE);
				return;
			}
			
			Dist<double[]> fDist = new EuclideanDist(fa);
			Dist<double[]> gDist = ga.length > 0 ? new EuclideanDist(ga) : null;
			
			ApplyExistingDialog aed = new ApplyExistingDialog(this, "Apply existing", true);
			if( aed.isOkPressed() ) {
				
				Grid2D<double[]> grid = null;
				List<double[]> l = null;
				try {
					grid = SomUtils.loadGrid(new FileInputStream(aed.getGridFile()));
					l = DataUtils.readCSV( new FileInputStream(aed.getMapFile()));
					
				} catch (FileNotFoundException e1) {
					e1.printStackTrace();
				}
				Map<GridPos,Set<double[]>> bmus = new HashMap<GridPos,Set<double[]>>();
				for( double[] d : l ) {
					GridPos p = new GridPos((int)d[0],(int)d[1]);
					if( !bmus.containsKey(p))
						bmus.put(p,new HashSet<double[]>());
					bmus.get(p).add(samples.get((int)d[2]));
				}
								
				SOMResultPanel srp = new SOMResultPanel(this, origData, samples, bmus, grid, fDist, gDist, ga);
				tp.addTab("Results (SOM," + (numSom++) + ")", srp);
				tp.setSelectedComponent(srp);
				tp.setTabComponentAt(tp.indexOfComponent(srp), new ButtonTabComponent(tp));
			}
			
		} else if (e.getPropertyName().equals(AnnPanel.TRAIN_PROP) && (Boolean) e.getNewValue()) {
			int[] fa = dataPanel.getFA();
			int[] ga = dataPanel.getGA(false);

			List<double[]> samples = dataPanel.getNormedSamples();
			SpatialDataFrame origData = dataPanel.getSpatialData();
			
			if (dataPanel.getSpatialData() == null) {
				JOptionPane.showMessageDialog(this, "No data loaded yet!", "No data!", JOptionPane.ERROR_MESSAGE);
				return;
			}
			
			int sampleLength = samples.get(0).length;
			Dist<double[]> fDist = new EuclideanDist(fa);
			Dist<double[]> gDist = ga.length > 0 ? new EuclideanDist(ga) : null;
			int t_max = annPanel.getNumTraining();
			
			Map<double[],Map<double[],Double>> distanceMatrix = null;
			if( annPanel.tpContextModel.getSelectedComponent() == annPanel.wmcPanel ) {
				try {
					File dmFile = ((WMCPanel)annPanel.wmcPanel).getDistMapFile();
					distanceMatrix = GeoUtils.readDistMatrixKeyValue(samples, dmFile);
				} catch( Exception ex ) {
					JOptionPane.showMessageDialog(this, "Could read/parse distance matrix file: "+ex.getLocalizedMessage(), "Read/Parse error!", JOptionPane.ERROR_MESSAGE);
					return;
				}
			}
						
			setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));
			for (int run = 0; run < annPanel.getRuns(); run++) {

				if (annPanel.tpANN.getSelectedComponent() == annPanel.somPanel) { // som
					SOMPanel sp = (SOMPanel) annPanel.somPanel;

					Grid2D<double[]> grid = ((SOMPanel) annPanel.somPanel).getGrid();
					SomUtils.initRandom(grid, samples);
					Map<GridPos, Set<double[]>> bmus;

					if (annPanel.tpContextModel.getSelectedComponent() == annPanel.wmcPanel) {
						WMCPanel wp = (WMCPanel) annPanel.wmcPanel;

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
						for (double[] d : samples)
							bmuHist.put(d, prototypes.get(r.nextInt(prototypes.size())));
						
						SorterWMC s = new SorterWMC(bmuHist, distanceMatrix , fDist, wp.getAlpha(), wp.getBeta());
						BmuGetter<double[]> bg = new SorterBmuGetterContext(s);
						ContextSOM som = new ContextSOM(sp.getKernelFunction(), sp.getLearningRate(), grid, (BmuGetterContext) bg, sampleLength);

						s.setHistMutable(true);
						for (int t = 0; t < t_max; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							som.train((double) t / t_max, x);
						}
						s.setHistMutable(false);

						bmus = SomUtils.getBmuMapping(samples, grid, bg);
					} else {
						BmuGetter<double[]> bg = null;

						if (annPanel.tpContextModel.getSelectedComponent() == annPanel.cngPanel) {
							bg = new SorterBmuGetter<double[]>(new KangasSorter<double[]>(gDist, fDist, ((CNGPanel) annPanel.cngPanel).getSNS()));
						} else if (annPanel.tpContextModel.getSelectedComponent() == annPanel.geoSomPanel) {
							bg = new KangasBmuGetter<double[]>(gDist, fDist, ((GeoSOMPanel) annPanel.geoSomPanel).getRadius());
						} else if (annPanel.tpContextModel.getSelectedComponent() == annPanel.weightedPanel) {
							double w = ((WeightedPanel) annPanel.weightedPanel).getAlpha();
							Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
							map.put(fDist, 1 - w);
							map.put(gDist, w);
							Dist<double[]> wDist = new WeightedDist<double[]>(map);
							bg = new DefaultBmuGetter<double[]>(wDist);
						} else if (annPanel.tpContextModel.getSelectedComponent() == annPanel.augmentedPanel) {
							double a = ((AugmentedPanel) annPanel.augmentedPanel).getAlpha();
							Dist<double[]> aDist = new AugmentedDist(ga, fa, a);
							bg = new DefaultBmuGetter<double[]>(aDist);
						} else {
							bg = new DefaultBmuGetter<double[]>(fDist);
						}

						SOM som = new SOM(sp.getKernelFunction(), sp.getLearningRate(), grid, bg);
						for (int t = 0; t < t_max; t++) {
							double[] x = samples.get(r.nextInt(samples.size()));
							som.train((double) t / t_max, x);
						}
						bmus = SomUtils.getBmuMapping(samples, grid, bg);
					}

					log.debug("SOM nr: " + numSom);

					Map<double[], Set<double[]>> mapping = new HashMap<double[], Set<double[]>>();
					for (GridPos p : bmus.keySet())
						mapping.put(grid.getPrototypeAt(p), bmus.get(p));
					double qe = DataUtils.getMeanQuantizationError(mapping, fDist);
					log.debug("qe (fDist): " + qe);
					if (gDist != null) {
						double ge = DataUtils.getMeanQuantizationError(mapping, gDist);
						log.debug("qe (gDist): " + ge + " (" + qe / ge + ")");
					}

					SOMResultPanel srp = new SOMResultPanel(this, origData, samples, bmus, grid, fDist, gDist, ga);
					tp.addTab("Results (SOM," + (numSom++) + ")", srp);
					tp.setSelectedComponent(srp);
					tp.setTabComponentAt(tp.indexOfComponent(srp), new ButtonTabComponent(tp));

				} else { // ng
					NGPanel np = (NGPanel) annPanel.ngPanel;

					NG ng = null;
					Sorter<double[]> s = null;
					if (annPanel.tpContextModel.getSelectedComponent() == annPanel.wmcPanel) {
						WMCPanel wp = (WMCPanel) annPanel.wmcPanel;

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < np.numNeurons(); i++) {
							double[] rs = samples.get(r.nextInt(samples.size()));
							double[] d = Arrays.copyOf(rs, rs.length * 2);
							for (int j = rs.length; j < d.length; j++)
								d[j] = r.nextDouble();
							neurons.add(d);
						}

						Map<double[], double[]> bmuHist = new HashMap<double[], double[]>();
						for (double[] d : samples)
							bmuHist.put(d, neurons.get(r.nextInt(neurons.size())));

						s = new SorterWMC(bmuHist, distanceMatrix, fDist, wp.getAlpha(), wp.getBeta());
						ng = new ContextNG(neurons, np.getNeighborhoodRate(), np.getAdaptationRate(), (SorterWMC) s);

					} else {
						if (annPanel.tpContextModel.getSelectedComponent() == annPanel.cngPanel) {
							s = new KangasSorter<double[]>(gDist, fDist, ((CNGPanel) annPanel.cngPanel).getSNS());
						} else if (annPanel.tpContextModel.getSelectedComponent() == annPanel.weightedPanel) {
							double w = ((WeightedPanel) annPanel.weightedPanel).getAlpha();
							Map<Dist<double[]>, Double> map = new HashMap<Dist<double[]>, Double>();
							map.put(fDist, 1 - w);
							map.put(gDist, w);
							Dist<double[]> wDist = new WeightedDist<double[]>(map);
							s = new DefaultSorter<double[]>(wDist);
						} else if (annPanel.tpContextModel.getSelectedComponent() == annPanel.augmentedPanel) {
							double a = ((AugmentedPanel) annPanel.augmentedPanel).getAlpha();
							Dist<double[]> aDist = new AugmentedDist(ga, fa, a);
							s = new DefaultSorter<double[]>(aDist);
						} else
							s = new DefaultSorter<double[]>(fDist);

						List<double[]> neurons = new ArrayList<double[]>();
						for (int i = 0; i < np.numNeurons(); i++) {
							double[] rs = samples.get(r.nextInt(samples.size()));
							neurons.add(Arrays.copyOf(rs, rs.length));
						}
							
						ng = new NG(neurons, np.getNeighborhoodRate(), np.getAdaptationRate(), s);
					}

					if (s instanceof SorterWMC)
						((SorterWMC) s).setHistMutable(true);

					for (int t = 0; t < t_max; t++) {
						double[] x = samples.get(r.nextInt(samples.size()));
						ng.train((double) t / t_max, x);
					}

					if (s instanceof SorterWMC)
						((SorterWMC) s).setHistMutable(false);

					Map<double[], Set<double[]>> bmus = NGUtils.getBmuMapping(samples, ng.getNeurons(), s);

					// build graph, maybe this should better got to NGResultPanel
					Graph<double[], double[]> g = new UndirectedSparseGraph<double[], double[]>();
					for (double[] x : samples) {
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

					log.debug("NG nr: " + numNg);
					log.debug("vcount: " + g.getVertexCount());
					log.debug("ecount: " + g.getEdgeCount());
					double qe = DataUtils.getMeanQuantizationError(bmus, fDist);
					log.debug("qe (fDist): " + qe);
					if (gDist != null) {
						double ge = DataUtils.getMeanQuantizationError(bmus, gDist);
						log.debug("qe (gDist): " + ge + " (" + qe / ge + ")");
					}

					NGResultPanel srp = new NGResultPanel(this, origData, samples, bmus, g, fDist, gDist, ga);
					tp.addTab("Results (NG, " + (numNg++) + ")", srp);
					tp.setSelectedComponent(srp);
					tp.setTabComponentAt(tp.indexOfComponent(srp), new ButtonTabComponent(tp));
				}
			}
			setCursor(Cursor.getDefaultCursor());
		} else if (e.getPropertyName().equals(DataPanel.TRAIN_ALLOWED_PROP) ) {
			tp.setEnabledAt(1, (Boolean)e.getNewValue() );
		} else if (e.getPropertyName().equals(DataPanel.COORD_CHANGED_PROP)) {
			annPanel.setEnabledCentroidModels((Boolean)e.getNewValue());
		}  
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == aboutItem) {
			JOptionPane.showMessageDialog(this, "Spatial Analysis with (Self-Organinzing) Neural Networks (SPAWNN) ©2013-2015\n\n" + "All rights reserved.\n\n" + "Julian Hagenauer", "About", JOptionPane.INFORMATION_MESSAGE);
		} else if (e.getSource() == quitItem) {
			System.exit(1);
		} 
	}
}
