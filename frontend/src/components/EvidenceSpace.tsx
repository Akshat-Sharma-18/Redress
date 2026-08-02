import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Audit, Evidence, SubClaim } from "../types";

/** The 3D evidence space: retrieved clauses as nodes suspended in the dark.
 *
 *  This is a *spatial index*, not a replacement for the evidence list. The
 *  text lives in the 2D panel where it can be read, selected, and screen-read;
 *  what the 3D view adds is the shape of the retrieval — how many clauses came
 *  back, which source each came from, and which of them the verdict rests on,
 *  all visible at once without scrolling.
 *
 *  Deliberately small in scope: one draw call per node, no post-processing, no
 *  orbit controls. A demo that stutters is worse than no demo, and every
 *  frame here has to survive a laptop running a 27B model on the CPU beside
 *  it.
 */

interface Props {
  audit: Audit;
  active: SubClaim | null;
  onSelect: (id: string | null) => void;
}

const KIND_COLOR: Record<Evidence["source_kind"], number> = {
  policy: 0x8fb6d9,
  statute: 0xb39ede,
  precedent: 0xe3a23a,
  denial: 0xc4485e,
};

const TONE_COLOR: Record<SubClaim["tone"], number> = {
  contradicted: 0xc4485e,
  verified: 0x2bb3a3,
  contested: 0xe3a23a,
  pending: 0xe3a23a,
};

export function EvidenceSpace({ audit, active, onSelect }: Props) {
  const mount = useRef<HTMLDivElement>(null);
  // Mutable scene handles kept outside React state: these change every frame
  // and re-rendering the tree at 60fps would be pointless work.
  const api = useRef<{
    dispose: () => void;
    update: (audit: Audit, active: SubClaim | null) => void;
  } | null>(null);

  useEffect(() => {
    const el = mount.current;
    if (!el) return;
    const host: HTMLDivElement = el;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(0, 0, 8.6);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "low-power",
    });
    renderer.setClearColor(0x000000, 0);
    // Capped at 2: retina laptops report 3+, which quadruples fragment work
    // for a difference nobody can see on nodes this size.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const key = new THREE.DirectionalLight(0xc9a24b, 0.7);
    key.position.set(3, 4, 6);
    scene.add(key);

    const group = new THREE.Group();
    scene.add(group);

    const nodes: {
      id: string;
      mesh: THREE.Mesh;
      halo: THREE.Mesh;
      home: THREE.Vector3;
    }[] = [];
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let hovered: string | null = null;

    const sphere = new THREE.SphereGeometry(0.58, 22, 22);
    const ring = new THREE.RingGeometry(0.78, 0.9, 30);

    function build(a: Audit) {
      for (const n of nodes) {
        group.remove(n.mesh, n.halo);
        (n.mesh.material as THREE.Material).dispose();
        (n.halo.material as THREE.Material).dispose();
      }
      nodes.length = 0;

      const count = a.evidence.length;
      a.evidence.forEach((item, i) => {
        // Fibonacci-ish spiral on a sphere: spreads nodes evenly without the
        // clustering a naive lat/long grid produces at the poles.
        const t = count === 1 ? 0.5 : i / (count - 1);
        const phi = Math.acos(1 - 2 * (t * 0.8 + 0.1));
        const theta = Math.PI * (1 + Math.sqrt(5)) * i;
        const r = 3.1;
        const home = new THREE.Vector3(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.cos(phi) * 0.62,
          r * Math.sin(phi) * Math.sin(theta),
        );

        const mesh = new THREE.Mesh(
          sphere,
          new THREE.MeshStandardMaterial({
            color: KIND_COLOR[item.source_kind],
            roughness: 0.55,
            metalness: 0.15,
            transparent: true,
            opacity: 0.5,
          }),
        );
        mesh.position.copy(home);
        mesh.userData.id = item.id;

        const halo = new THREE.Mesh(
          ring,
          new THREE.MeshBasicMaterial({
            color: 0xc9a24b,
            transparent: true,
            opacity: 0,
            side: THREE.DoubleSide,
          }),
        );
        halo.position.copy(home);

        group.add(mesh, halo);
        nodes.push({ id: item.id, mesh, halo, home });
      });
    }

    function applyState(a: Audit, sub: SubClaim | null) {
      const cited = new Set(sub?.citations.map((c) => c.chunk_id) ?? []);
      const tone = sub ? TONE_COLOR[sub.tone] : 0xc9a24b;
      for (const n of nodes) {
        const item = a.evidence.find((e) => e.id === n.id);
        const isCited = cited.has(n.id);
        const mat = n.mesh.material as THREE.MeshStandardMaterial;
        mat.color.setHex(
          isCited ? tone : KIND_COLOR[item?.source_kind ?? "policy"],
        );
        mat.opacity = sub ? (isCited ? 1 : 0.16) : 0.5;
        mat.emissive.setHex(isCited ? tone : 0x000000);
        mat.emissiveIntensity = isCited ? 0.5 : 0;
        (n.halo.material as THREE.MeshBasicMaterial).opacity = isCited ? 0.75 : 0;
        n.mesh.scale.setScalar(isCited ? 1.25 : 1);
        n.halo.scale.setScalar(isCited ? 1.25 : 1);
      }
    }

    function resize() {
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    const onPointerMove = (e: PointerEvent) => {
      const rect = host.getBoundingClientRect();
      pointer.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
    };
    const onClick = () => onSelect(hovered);

    host.addEventListener("pointermove", onPointerMove);
    host.addEventListener("click", onClick);
    const observer = new ResizeObserver(resize);
    observer.observe(host);

    let raf = 0;
    let running = true;
    const clock = new THREE.Clock();
    // Respect the preference by stopping the orbit, not by removing the
    // view. The information here -- how many clauses, from which sources,
    // which are cited -- has nothing to do with movement, and hiding it
    // would take that away from the people who asked for less motion.
    const still = window.matchMedia("(prefers-reduced-motion: reduce)");

    function frame() {
      if (!running) return;
      raf = requestAnimationFrame(frame);
      // Clamped: requestAnimationFrame is paused while the tab is hidden,
      // and Clock keeps accumulating. Returning to the tab after a minute
      // would otherwise deliver a 60-second delta and spin the scene through
      // several revolutions in one frame.
      const dt = Math.min(clock.getDelta(), 0.05);

      // A slow orbit: alive, not distracting. Framed as rotation of the
      // whole group rather than the camera so the lighting stays fixed and
      // nodes read as objects rather than a spinning image.
      if (!still.matches) group.rotation.y += dt * 0.11;

      for (const n of nodes) {
        // Billboard the halos so they always face the viewer.
        n.halo.quaternion.copy(camera.quaternion);
        n.halo.position.copy(n.mesh.position);
      }

      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(nodes.map((n) => n.mesh))[0];
      const nextHover = (hit?.object.userData.id as string) ?? null;
      if (nextHover !== hovered) {
        hovered = nextHover;
        host.style.cursor = hovered ? "pointer" : "default";
      }

      renderer.render(scene, camera);
    }

    build(audit);
    applyState(audit, active);
    resize();
    // The synchronous resize above runs before layout on first mount, where
    // clientWidth/Height are still 0 and it bails. One more on the next frame
    // catches the laid-out size; the ResizeObserver handles everything after.
    requestAnimationFrame(resize);
    frame();

    api.current = {
      update: (a, sub) => {
        build(a);
        applyState(a, sub);
      },
      dispose: () => {
        running = false;
        cancelAnimationFrame(raf);
        observer.disconnect();
        host.removeEventListener("pointermove", onPointerMove);
        host.removeEventListener("click", onClick);
        // WebGL contexts are a limited resource: a page that mounts this
        // repeatedly without disposing will eventually lose the context and
        // render nothing, with no error in the console.
        sphere.dispose();
        ring.dispose();
        for (const n of nodes) {
          (n.mesh.material as THREE.Material).dispose();
          (n.halo.material as THREE.Material).dispose();
        }
        renderer.dispose();
        renderer.domElement.remove();
      },
    };

    return () => {
      api.current?.dispose();
      api.current = null;
    };
    // Built once; content changes go through `update` below rather than
    // tearing down and rebuilding the WebGL context on every selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    api.current?.update(audit, active);
    // Mirrors the applied state onto the DOM so the scene can be asserted on
    // from outside WebGL: readPixels cannot distinguish "did not re-render"
    // from "re-rendered identically", and a canvas is otherwise a black box
    // to any test.
    if (mount.current) {
      mount.current.dataset.cited = String(active?.citations.length ?? 0);
      mount.current.dataset.nodes = String(audit.evidence.length);
    }
  }, [audit, active]);

  return <div className="evidence-space" ref={mount} aria-hidden="true" />;
}
