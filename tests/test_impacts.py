import xobjects as xo
import xtrack as xt
import xpart as xp
import xcoll as xc


def test_impacts():
     for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        elements = [
                    xt.Multipole(knl=[0, 0.021], _context=context),
                    xt.Drift(length=1.35, _context=context),
                    xt.Drift(length=0, _context=context), # H collimator
                    xt.Drift(length=1, _context=context),
                    xt.Drift(length=0, _context=context), # V collimator
                    xt.Drift(length=3.78, _context=context),
                    xt.Multipole(knl=[0, -0.013], _context=context),
                    xt.Drift(length=5, _context=context),
                    xt.Cavity(frequency=400e7, voltage=6e6, _context=context),
                    xt.Drift(length=.5, _context=context),
                    xt.Drift(length=0, _context=context), # skew 1 collimator
                    xt.Drift(length=1, _context=context),
                    xt.Drift(length=0, _context=context), # skew 2 collimator
                    xt.Drift(length=5, _context=context),
                    xt.Multipole(knl=[0, -0.0078], _context=context),
                    xt.Drift(length=4.2, _context=context),
                ]

        line = xt.Line(elements=elements, _context=context)
        coll_manager = xc.CollimatorManager(line=line, record_impacts=True)
        coll_manager.add()  # TODO
        coll_manager.install_black_absorbers()
        coll_manager.align_collimators_to('front')
        coll_manager.build_tracker()
        coll_manager.set_openings()
        n_sigmas = 10
        n_part = 5000
        x_norm = np.random.uniform(-n_sigmas, n_sigmas, n_part)
        y_norm = np.random.uniform(-n_sigmas, n_sigmas, n_part)
        part = xp.build_particles(tracker=coll_manager.tracker, x_norm=x_norm, y_norm=y_norm,
                                  scale_with_transverse_norm_emitt=(3.5e-6, 3.5e-6),
                                 )
        coll_manager.track(part, num_turns=10)
        part.reshuffle()
        # All particles in the impact table are lost on a collimator
        states = np.unique(part.state[coll_manager.impacts[('parent','id')].values])
        assert len(states)==1 and states[0]==-333
        # All particles that are lost have exactly the same id's as those in the table
        assert np.array_equal(part.particle_id[part.state<1], coll_manager.impacts[('parent','id')].values)
        # And the same coordinates
        assert np.allclose(part.x[part.state<1],      coll_manager.impacts[('parent','x')].values, atol=1e-15, rtol=0)
        assert np.allclose(part.px[part.state<1],     coll_manager.impacts[('parent','px')].values, atol=1e-15, rtol=0)
        assert np.allclose(part.y[part.state<1],      coll_manager.impacts[('parent','y')].values, atol=1e-15, rtol=0)
        assert np.allclose(part.py[part.state<1],     coll_manager.impacts[('parent','py')].values, atol=1e-15, rtol=0)
        assert np.allclose(part.zeta[part.state<1],   coll_manager.impacts[('parent','zeta')].values, atol=1e-15, rtol=0)
        assert np.allclose(part.delta[part.state<1],  coll_manager.impacts[('parent','delta')].values, atol=1e-15, rtol=0)
        assert np.allclose(part.energy[part.state<1], coll_manager.impacts[('parent','energy')].values, atol=1e-15, rtol=0)

