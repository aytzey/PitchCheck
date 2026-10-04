# PitchServer kod, sağlamlık ve hız incelemesi — 4 Ekim 2026

`origin/main` üzerindeki en güncel commit (`cef084d`, 23 Eylül 2026) temel alındı. Değişiklikler `fix/server-runtime-hardening` branch'inde. Sunucudaki mevcut PitchServer dağıtımı güncellendi; diğer servislerin konteyner kimlikleri, başlangıç zamanları, restart sayıları ve sağlık durumları başlangıç kaydıyla karşılaştırıldı.

## Gerçek sunucu ve dağıtım

| Alan | Doğrulanan değer |
|---|---|
| Sunucu | `dkmserver@78.186.120.189`, SSH `2022` |
| Dizin | `/home/dkmserver/Desktop/Machinity/aytug/pitchserver` |
| GPU | NVIDIA RTX 5080; CUDA'nın gördüğü toplam bellek 15,469 GiB |
| CPU / RAM | 20 mantıksal CPU / yaklaşık 64 GB RAM |
| Paylaşılan yük | PitchServer dahil 166 çalışan konteyner |
| Compose / servis | `pitchserver` / `tribe` |
| Konteyner | `pitchserver_tribe` |
| Mevcut erişim | `127.0.0.1:18090`, `https://pitchserver.machinity.ai` |
| Son dağıtılan imaj; gerçek TRIBE aday seçimi dahil | `sha256:942421b4faa73f9788f61e1d75e704ef3aac84c9e399ade79a709fc42e439eb0` |
| Önceki sağlamlık / performans imajı | `sha256:9572265cf67a3be3186ff479552126aa370f06deb93240a7563470d010e50721` |

Traefik, diğer Compose projeleri, GPU sürücüsü ve sistem paketleri değiştirilmedi. İşlem öncesinde yaklaşık 4 GiB GPU belleği başka süreçlerce kullanılıyordu. Sunucunun swap alanı doluydu; PitchServer'ın swap kullanması engellendi.

## Akış ve bulunan sorunlar

Sunucuya dağıtılan imaj Python backend'idir. Masaüstü ve web değişiklikleri bu branch üzerinde build/test edildi; hazır native geliştirme build'i `src-tauri/target/debug/pitchcheck-desktop` yolundadır.

Masaüstü Tauri istemcisi SSH tüneli ve kullanıcı oturumu üzerinden; mobil istemci HTTP API üzerinden çalışıyor. Python API metni kelime olaylarına çeviriyor, Hermes metin özelliklerini çıkarıyor, TRIBE ile 20.484 voxel tahmini oluşturuyor, sayısal özellikleri özetliyor ve mevcut OpenRouter modeliyle raporu tamamlıyor. Doğrudan metin modunda TTS/WhisperX çalışmıyor. Son takip çalışmasında `/refine`, üç farklı LLM taslağını ve orijinali gerçek TRIBE ile ölçer; olgusal/bağlamsal kontrolü geçen ölçülmüş adaylar arasından mevcut nöral/semantik ağırlıklarla seçim yapar. Seçimden sonra ölçülmemiş yeni bir metin yazılmaz.

| Sorun | Düzeltilen davranış ve kanıt |
|---|---|
| İstemci isteği iptal edilince GPU kilidi erken bırakılıyordu | İşçi kendi izin hakkını taşıyor. HTTP iptali veya zaman aşımı sonrasında çalışan thread tamamlanana kadar yeni GPU işi ve model boşaltma engelleniyor. İptal ve timeout regresyonları geçti. |
| LLM çağrıları sınırsız thread işi oluşturabiliyordu | Skor yorumlama ve refine ortak iki işlem sınırını kullanıyor. Kuyruk 5 saniyede `429`, istemciye yönelik LLM süresi 180 saniyede `504` veriyor. İşçi arka planda sürerse izin hakkı korunuyor. |
| Doğrudan API ve web refine için parçalı büyük gövdeler sınırsız okunabiliyordu | Python API ve iki Next.js rotası 128 KiB gerçek byte sınırını uyguluyor. Gövde okuma süresi 15 saniye. Parçalı büyük istek `413`, duran gövde `408` testleri geçti. |
| Giriş/parola doğrulama kilitleri event loop'u durdurabiliyordu | Parola işlemleri ve auth dependency framework thread havuzunda çalışıyor. Skor sırasında sağlık yanıtları canlı ölçüldü. |
| Oturum tablosu giriş yapıldıkça büyüyordu | Süresi geçmiş oturumlar yeni girişte temizleniyor; en fazla 128 oturum tutuluyor. Limit dolunca en eski oturum çıkarılıyor. |
| Üst kütüphanenin sardığı CUDA OOM hatası fallback'i atlıyordu | Hata nedenleri zinciri inceleniyor. Sarmalanmış OOM için Accelerate/CPU kurtarma regresyonu geçti. Accelerate bütçesi toplam GPU belleği yerine kullanılabilir bellekten hesaplanıyor. |
| Metin modeli her yeni metinde yeniden yükleniyordu | TRIBE'ın üst kütüphanedeki koşulsuz encoder boşaltması, mevcut `TRIBE_UNLOAD_TEXT_MODEL_AFTER_SCORE` politikasıyla uyumlu hale getirildi. Encoder işlem aralarında korunuyor. |
| Her kelime için bütün bağlamın bütün katmanları CPU'ya kopyalanıyordu | Yalnızca hedef kelime token'ları CPU'ya aktarılıyor. Token/katman birleştirme aritmetiği aynı. Gerçek Torch testi padding ve çok token'lı hedefleri kapsıyor. |
| Kelime özellikleri RAM ve diskte birikiyordu | Kütüphanenin kendi feature-cache temizleme işlemi her tahminin `finally` bloğunda çalışıyor. Model ağırlıkları korunuyor; sınırlı tahmin önbelleği kullanılıyor. GPU karşılaştırmasında her işlemden sonra feature-cache kayıt sayısı sıfır. |
| Checkpoint'teki eğitim ayarı 20 veri yükleyici süreci açıyordu | İnferans `num_workers=0` kullanıyor; konteynerde gereksiz süreç ve bellek kopyaları oluşmuyor. |
| Offline modda hazır ağırlıklar olmasına rağmen model adı doğrulaması internete gidiyordu | Yalnızca yerelde config dosyası bulunan model adı offline doğrulamada kabul ediliyor. Offline, salt okunur ağırlık mount'u ile soğuk inferans geçti. |
| Hata logları müşteri metni veya sağlayıcı hata gövdesi içerebiliyordu | Loglar hata sınıfı / HTTP durumuyla sınırlı. Gizli metin içeren hata regresyonunda müşteri içeriği loga düşmüyor. |
| Masaüstü bağlantısı sunucu Compose/env dosyalarını yazıyor ve 15 dakikalık `latest` güncellemesi kuruyordu | Bağlantı mevcut dağıtımı kullanıyor; durmuşsa yalnızca `tribe` servisini yerel imajla başlatıyor. Bağlantı dosya veya cron yazmıyor. Başarısız girişte SSH tüneli kapatılıyor. Sunucudaki eski otomatik güncelleme kaldırıldı. |
| Web health backend kapalıyken başarılı görünüyordu | GET health backend kapalıyken `ok:false` ve `503` döndürüyor. HEAD hafif web liveness kontrolü olarak korunuyor. |

## Kaynak ve işletim sınırları

| Kaynak / koruma | Uygulanan değer |
|---|---|
| CPU | 4 CPU kota; OMP/MKL/OpenBLAS için 2 thread |
| RAM | 12 GiB konteyner limiti |
| Swap | 0 ilave swap |
| GPU | PyTorch allocator için 8 GiB; GPU `0` |
| Süreç | 256 PID |
| Paralel GPU işi | 1 |
| Paralel LLM işi | 2 |
| Tahmin önbelleği | En fazla 8 kayıt / toplam 64 MiB |
| Kullanılmayan model | 600 saniye sonra boşaltma |
| Log | 10 MB × 3 dosya döndürme |
| Kapanış | İşçiler tamamlanmadan model boşaltılmıyor; konteyner durma bütçesi 40 saniye |
| Süreç ayrıcalıkları | `no-new-privileges`; tüm capability'ler kaldırılmış, yalnızca karışık sahiplikteki model/auth dosyaları için `DAC_OVERRIDE`, `FOWNER` korunmuş |
| Dağıtım | Yerel imaj ID'si sabit, `pull_policy: never`, `--no-deps tribe` |

Ortak üst dizinin `.env` dosyası artık servise aktarılmıyor. Mevcut OpenRouter ve auth ayarları ayrı `runtime.env` dosyasında, `0600` izinleriyle korunuyor. Mevcut auth dosyası ve kullanıcı parolası değiştirilmedi. Sunucunun Compose dosyası `0444`; eski masaüstü sürümlerinin bağlantı sırasında dosyayı yanlışlıkla yeniden yazması engelleniyor. Yeni masaüstü kodu bu dosyaya yazmadan bağlanır; eski provisioning yapan sürümler güncellenmelidir.

## Ölçülmüş hız

Eski imaj ve son kod, aynı RTX 5080 üzerinde sırayla; 4 CPU, 12 GiB RAM, 8 GiB allocator bütçesi, batch=4 ve boş ayrı feature dizinleriyle çalıştırıldı. Hugging Face ağırlıkları diskten ve offline yüklendi. Exca'nın rastgele feature iş sırası için karşılaştırma seed'i sabitlendi. Metinler sentetik; müşteri verisi kullanılmadı. Son temizleme düzeltmesi dahil ölçümler:

| Girdi | Eski imaj | Yeni kod | Azalma |
|---|---:|---:|---:|
| 42 kelime; model soğuk | 4,8947 sn | 4,2963 sn | %12,2 |
| 126 kelime; farklı yeni metin, model sıcak | 4,6019 sn | 2,3823 sn | %48,2 |
| 378 kelime; farklı yeni metin, model sıcak | 24,7823 sn | 15,4330 sn | %37,7 |
| Aynı 126 kelimelik metin; tahmin önbelleği | 1,6 ms | 0,8 ms | GPU tekrar çalışmıyor |

Üç boyut için tüm voxel tahmin matrislerinde maksimum mutlak fark **0,0**. GPU'da ölçülen tepe PyTorch allocation **7,022 GiB**; yeni süreç için ölçülen tepe RSS yaklaşık **5.965 MiB**. Bunlar bu iş yükünün ölçümleridir; tüm metinler için performans garantisi veya birden fazla tekrarın istatistiksel medyanı değildir.

Batch=8 denemesi aynı sayısal tahminleri korumadı; kullanılmadı. Exca eksik feature işlerini rastgele sıraya koyduğu için soğuk feature-cache ve farklı batch bileşimleri önceki sürümde de farklı sayısal çıktılar oluşturabiliyor. Parite testi aynı sıralama/seed altında yapılır; LLM yanıtlarının deterministik olduğu iddia edilmez.

## ONNX / TensorRT kararı

Bu dağıtım mevcut PyTorch CUDA akışını kullanıyor. Ölçülen darboğazlar gereksiz encoder yüklemesi ve tam bağlamın CPU'ya taşınmasıydı; mevcut ağırlık ve aritmetik korunarak giderildi. İlk dağıtımda ONNX/TensorRT dönüşümü uygulanmadı.

Takip çalışmasında 28 encoder bloğu TensorRT'ye dönüştürülüp gerçek TRIBE tahminleri üretildi. Sıcak encoder ile 126 kelimede native **2,3397 sn**, TensorRT **2,4836 sn** ölçüldü; çıktı eşitliği testi geçmedi. 378 kelimelik TensorRT işi 8 GiB süreç GPU bütçesini aşınca yalnızca deneme konteyneri durduruldu. Canlı imaj korunuyor. Ayrıntılar, kaynak sınırları ve tekrar üretme yolu [TensorRT deneme raporunda](tensorrt-experiment.md).

Ayrıca tam API süresinde dış LLM çağrısı önemli yer tutuyor: ilk sağlamlık dağıtımında soğuk skor+rapor **17,060 sn**, refine **19,939 sn** ölçüldü. O imajın otomatik restart testi sonrası **18,549 sn** ve **15,495 sn** ölçüldü. Bu ilk ölçümde mevcut `google/gemini-3.8-flash` ve refine kalite adımları korundu. Encoder'ın TensorRT'ye taşınması bu dış çağrının süresini azaltmaz. TensorRT seçilirse dinamik şekiller için profil ve bellek bütçelerinin yeniden doğrulanması gerekir; bu NVIDIA'nın [dinamik şekil dokümanında](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dynamic-shapes-basics.html) anlatılır.

## İlk sağlamlık dağıtımının doğrulaması ve kapsamı

- Sabit GPU imajında **130 Python testi geçti**. Hafif CPU/CI ortamında **129 geçti, 1 Torch/neuralset testi atlandı**; o test gerçek GPU bağımlılıklarının bulunduğu imajda geçti.
- **49 web testi**, **9 Rust testi**, ESLint, Next.js üretim build'i, desktop static build ve native desktop build geçti.
- Python mock testleri CI'ye eklendi. Test bağımlılıkları sunucuda doğrulanan sürümlere sabitlendi. Paylaşılan sunucu Compose dosyasına ayrı CI config kontrolü eklendi.
- Gerçek API: kullanıcı girişi, auth olmadan `401`, büyük istek `413`, geçersiz istek `422`, gerçek TRIBE skoru, gerçek sağlayıcıyla refine, işlem sırasında unload engeli ve işlem sonunda GPU boşaltma doğrulandı.
- İzole testte işlem sırasında en uzun health yanıtı **102 ms**; canlı ilk geçişte **469 ms**, son imajda **121 ms**, son imajın restart testi sonrası **457 ms** oldu.
- Yalnızca PitchServer'ın ana sürecine SIGTERM gönderilerek `unless-stopped` otomatik restart davranışı doğrulandı. Kalıcı auth dosyasıyla tekrar giriş, gerçek skor ve refine yeniden geçti. Sunucuya reboot uygulanmadı.
- Yeni masaüstü bağlantısının ürettiği SSH komutu canlı hostta çalıştırıldı; mevcut sabit imajı döndürdü ve dosya yazmadı. GUI üzerinden SSH/parola bağlantısı bu çalışmada ayrıca tıklanarak test edilmedi.
- Public HTTPS health yönetilen Firefox'ta görüntülendi; auth gerekli, model CUDA, bellek sınırı 8 GiB ve Rust sayısal çekirdek aktif görüldü. Görev tarayıcı oturumu kapatıldı.
- Başlangıç kaydıyla karşılaştırmada **diğer 165 servisin tamamı** aynı konteyner kimliği, başlangıç zamanı, restart sayısı ve sağlık durumuyla kaldı. Test konteynerleri kaldırıldı.

Kod incelemesi `ce-code-review` ve sadeleştirme kontrolü `ce-simplify-code` ölçütleriyle ana oturumda sırayla yapıldı. Kullanıcının AGENTS talimatı nedeniyle bağımsız alt ajan veya ikinci model incelemesi yapılmadı. İnceleme tamamlandı; kaynak, API sözleşmeleri, iptal/kapanış davranışı, auth, hata logları, cache ömrü ve dağıtımın tek servise sınırlandırılması kontrol edildi.

8 GiB sınırı PyTorch allocator'a aittir; bu GPU'da donanımsal ayrı GPU bölümü veya bağımsız GPU zamanlayıcı kotası değildir. Diğer GPU işlerinin gecikmesine sıfır etki garantisi verilmez. En büyük 30.000 karakterlik giriş ve aynı anda yoğun başka GPU işleri paylaşılan canlı makinede stres testine sokulmadı. In-memory auth oturumları restart'ta kaybolur; kullanıcı yeniden giriş yapar. Bu sınırlar çalışan imajın ve gerçek testlerin kapsamıdır.

## Tekrar çalıştırma ve geri dönüş

Yerel kontroller:

```bash
python -m pip install -r tribe_service/requirements-test.txt
python -m pytest tribe_service/tests -q
npm ci
npm run lint
npm test
npm run build
npm run build:desktop-web
cargo test --manifest-path src-tauri/Cargo.toml
cargo build --manifest-path src-tauri/Cargo.toml
```

Sunucudaki mevcut sürümü kontrollü yeniden başlatmak/başlatmak:

```bash
ssh -p 2022 dkmserver@78.186.120.189
cd /home/dkmserver/Desktop/Machinity/aytug/pitchserver
./update-pitchserver.sh
```

Bu komut imaj çekmez; yalnızca sabit mevcut `tribe` servisini uzlaştırır. Yeni bir sürüm için ayrı imaj oluşturulup test edilmeli, `.env` içindeki `PITCHSERVER_IMAGE` doğrulanan yeni yerel imaj ID'siyle değiştirilmelidir. Model/auth mount'ları ve shared Traefik korunmalıdır. `Dockerfile.server` doğrulanmış temel GPU imajına uygulama kodunu kopyalar; sürücü veya tüm bağımlılıkları tekrar kurmaz.

Son değişiklik öncesine dönüş:

```bash
/home/dkmserver/Desktop/Machinity/aytug/pitchserver/rollback-20261004T135434Z/rollback.sh
```

Çalışma başlamadan önceki orijinal servise dönüş:

```bash
/home/dkmserver/Desktop/Machinity/aytug/pitchserver/rollback-20261004T134614Z/rollback.sh
```

Rollback scriptleri yerel eski imajı ve kaydedilmiş Compose/env ayarlarını kullanır; yalnızca `tribe` servisini yeniler. Eski otomatik `latest` cron işi yeniden etkinleştirilmez. Yedeklerde sırlar bulunduğundan dizinler `0700`, dosyalar `0600` olarak sunucuda tutulur ve Git'e alınmaz.

Tekrar üretilebilir GPU karşılaştırması için `scripts/benchmark-runtime.py`, auth/API kabul kontrolü için `scripts/smoke-runtime.py` kullanılır. Karşılaştırma önce ayrı bir sınırlı konteynerde eski imajla `--output /audit/before`, ardından yeni imajla boş başka `TRIBE_CACHE_DIR` ve `--output /audit/after --compare /audit/before` ile sırayla çalıştırılmalıdır. Her iki koşuda aynı GPU/thread limitleri ve ortak read-only Hugging Face ağırlık mount'u kullanılmalıdır. `--compare` matris eşitliğini ve yeni sürümün işlem sonunda feature-cache'inin boş olmasını doğrular.

Sunucudaki ayrıntılı test çıktıları ve `.npy` matrisleri `audit-20261004/` altında; son performans kaydı `final/metrics.json`, servis karşılaştırması `service-verification.json`, dağıtım/rollback yolu `deployment.json` dosyalarında bulunur. Auth/API testi yalnızca durum, süre ve model adını kaydeder; parola veya token yazdırmaz.

## Taskfulight özel iOS entegrasyonu — önceki dağıtımın ölçümleri

Taskfulight'ın mevcut `feat/taskfulight-v2` branch'ine **Şimdi → İkna** ekranı eklendi. Ekran mevcut PitchServer hesabıyla HTTPS üzerinden bağlanır; giriş bilgileri ve token iPhone'un cihazda kalan, kilit açıkken erişilebilir Keychain kaydındadır. Kullanıcı bir dokunuşla modeli yükleyebilir veya boşaltabilir; analiz, iki turlu metin iyileştirme ve yeni metni yeniden analiz ederek karşılaştırma aynı ekrandadır. Seçilen model uygulamanın mevcut ayarından alınır: bu kabul testinde **`deepseek/deepseek-v4-flash`** kullanıldı. Sunucunun başka istemciler için varsayılan modeli değiştirilmedi.

Backend'de yeni auth gerektiren `POST /runtime/load`, TRIBE ile aynı cached Hermes encoder'ını tam yükler; sadece wrapper oluşturup hazır görünmez. Tekrar yükleme idempotenttir. Mevcut GPU işlem kilidi ve backpressure sınırları korunur. Unload isteği iptal edilse bile boşaltma thread'i bitene kadar pipeline kilidi tutulur. Yeni `POST /auth/logout` yalnızca arayan oturumu iptal eder; diğer oturumları korur.

Canlı testte DeepSeek'in varsayılan yüksek düşünme süresi public HTTPS sınırını aştı ve `524` oluştu. Yalnızca bu model için, operatör açık bir reasoning effort belirtmediyse, OpenRouter payload'ına `reasoning: {enabled: false}` eklenir. Hem skor raporu hem refine ortak politikayı kullanır. İkinci eleştirmen turu korunur. `minimal` denemesi aynı rapor tanılamasında **98,476 sn**, düşünme kapalı deneme **34,531 sn** sürdü; yalnız istemci timeout'unu artırmak edge timeout'unu çözmedi. Eksik `persuasion_score` alanıyla gelen JSON artık mevcut retry bütçesini kullanır; sessizce başarılı semantik rapor sayılmaz. Neural-only fallback uygulamada açıkça gösterilir.

Son imajda gerçek public HTTPS kabul sonucu; sentetik endüstriyel satış metni:

| İşlem / ölçüm | Sonuç |
|---|---:|
| Tam model yükleme | 4,783 sn |
| Tekrar yükleme | 0,124 sn |
| İlk analiz + gerçek DeepSeek raporu | 43,124 sn; skor 35 |
| Gerçek iki turlu refine | 9,528 sn |
| Yeni metnin ayrı gerçek analizi | 28,852 sn; skor 54 |
| Son native TRIBE hesabı; 27 kelime | 0,411 sn |
| İşlem sırasında en uzun health | 0,416 sn |
| Model boşaltma | 0,653 sn |
| Son analizde CUDA allocated / peak | 6,652 / 6,702 GiB |
| Model yüklü, işlem başlamadan RAM | 1,769 GiB |
| Model boşaltılmış RAM | 1,616 GiB |

RAM satırları Docker working-set ölçümüdür; yalnız yükleme/bekleme durumunu kapsar. Daha uzun gerçek inference için önceki performans karşılaştırmasında ölçülen tepe RSS yaklaşık 5.965 MiB idi. CUDA ölçümü GPU belleğidir ve RAM'den ayrıdır. Skor artışı bu tek örnekteki model değerlendirmesidir; gerçek kişilerin ikna olma olasılığı veya genel kalite garantisi değildir. Tam uygulama yolu fiziksel iPhone'da henüz doğrulanmadı; Flutter widget testleri ile native iOS/TestFlight CI ayrı kontrollerdir.

Son imajda **136 Python testi geçti**; yerel hafif ortamda 135 geçti, Torch/neuralset bağımlılık testi atlandı ve gerçek GPU imajında geçti. Auth olmadan load `401`, çalışan GPU işinde unload engeli, tam modelin sağlıkta hazır görünmesi, gerçek sağlayıcıyla iki turlu refine, gerçek yeniden skor, unload ve logout sonrası eski token'ın `401` alması doğrulandı. Public HTTPS health yönetilen Firefox'ta tekrar doğrulandı; görev oturumları kapatıldı. Test sonunda model boşaltıldı.

Diğer **165 kalıcı servisin** kimliği, başlangıç zamanı, restart sayısı ve sağlık durumu ilk Taskfulight snapshot'ıyla karşılaştırıldığında değişmedi. Son dağıtımın ham snapshot'ına eşzamanlı çalışan geçici pytest konteyneri de girmişti; `--rm` ile normal çıkışı ham karşılaştırmada bir silinme olarak görünür. Ham kayıt korundu; kalıcı servislerin ilk baseline'a göre ayrı doğrulaması `audit-taskfulight-20261004-final/service-verification.json`, açıklama `deployment-verification-note.json` dosyasındadır. GPU zamanında sıfır etki garantisi verilmez; mevcut kaynak sınırları korunur.

Kanıtlar sunucuda `audit-taskfulight-20261004-final/` altında `live-api.json`, `live-api.log`, `memory.json`, `deployment.json` ve `service-verification.json` dosyalarındadır. Sentetik test metni kayıtlıdır; parola/token/API anahtarı kayıtlı değildir. Son küçük backend değişikliği öncesine dönüş `rollback-20261004T180820Z/rollback.sh`; Taskfulight entegrasyonunun tümünü geri alıp önceki sağlamlık imajına dönüş `rollback-20261004T174806Z/rollback.sh` ile yapılır. Her iki script yalnız `tribe` servisini yeniler.


## Gerçek TRIBE aday araması — kullanıcı örneği sonrası son dağıtım

Önceki mobil refine yalnız dil modeliyle iki tur yazıyordu. Kullanıcının kişisel davet örneği, açıkça belirtilen hoşlanmama tercihinin göz ardı edildiğini ve verilmemiş kısa klip ayrıntısının yazıya eklendiğini gösterdi. Son akışta mevcut `deepseek/deepseek-v4-flash` üç farklı taslak üretir; orijinal dahil dört metnin her biri gerçek CUDA TRIBE modeliyle ölçülür. Eleştirmen her adayı doğruluk, asıl amaç, alıcı tercihi ve gönderenin dili açısından değerlendirir. Python, uygun adaylar arasında gerçek nöral kanıt ile semantik bağlamın mevcut ağırlıklı bileşimini kullanarak seçim yapar. Yüksek nöral puan yanlış bilgiyi meşrulaştıramaz; uygun iyileşme yoksa orijinal korunur.

Analizdeki örnek yeniden yazımlar artık taslak üretimine aktarılmaz; bunlar yeni gerçekler için kaynak değildir. Gerçek HTTPS ilk kabul denemesinde eleştirmenin önerideki verilmemiş cuma bilgisini onaylaması yakalandı. Tarih, sayı, klip/bilet ayrıntıları yalnız orijinal, persona ve gerçekten cevaplanmış bağlamda varsa kabul edilir. Sorulmuş ama cevaplanmamış soru olgusal kaynak sayılmaz. Doldurulmamış köşeli parantezli yer tutucular da elenir. Bu sınırlı Türkçe/İngilizce ayrıntı denetimi genel olgusal çıkarımın yerini tutmaz; kalan iddialar için semantik kontrol korunur. Başarısız veya eksik eleştirmen çıktısı ölçümsüz yeniden yazıma düşmek yerine redakte edilmiş hata verir.

GPU izin hakkı dört metnin bütün ölçüm süresi boyunca işçide kalır; iptal veya HTTP timeout bu hakkı erken bırakmaz. İşlem sırasında unload reddedilir. API yanıtındaki `tribe_guidance` orijinal ve üç adayın metnini, gerçek model/mode kimliğini, voxel/segment sayısını, nöral sinyallerini, ayrı bağlam puanını, seçimi ve elenme nedenlerini taşır. Taskfulight bu sözleşmeyi doğrulamadan gerçek TRIBE karşılaştırması iddiası göstermez. Uygulamadaki açılır aday karşılaştırması seçimin kanıtını görünür kılar; genel kanal kişisel davetlerde başlangıç seçimidir.

Eleştirmene gönderilen yinelenen eksen açıklamaları ve araştırma eki kaldırıldı; gerçek sayısal ölçümler korunuyor, tam eksen bilgisi API kanıtında kalıyor. Önceki ara sürümde dış sağlayıcı nedeniyle refine 104,159 sn sürdü. Bu bir hız garantisi değildir; gerçek model hesabı ile dış sağlayıcı gecikmesi ayrı ölçülür.

Son backend kaynak commit'i `52c2c7924159cb740fbd459136967f444c49b569`; `/app` altındaki 28 izlenen runtime dosyasının SHA-256 değeri kaynakla bire bir doğrulandı. GPU bağımlılık imajında **140 Python testi**, hafif yerel ortamda **139 test** geçti; yalnız gerçek Torch/neuralset testi yerelde atlandı ve imajda geçti. Karşıt testte aynı semantik değerlendirme tutulup nöral ölçümler değiştirilince kazanan aday değişiyor. Verilmemiş tarih/klip/bilet ve yer tutucu en yüksek model puanını alsa ve eleştirmen onaylasa bile eleniyor. İptal, eksik/çift aday, eksik eleştirmen ve özel hata içeriği regresyonları geçti.

Taskfulight kaynak commit'i `a85dcbb9c126bdc0120c834edf7d56573e90b350`. **136 Flutter testi**, analiz ve release hygiene geçti. Yeni imzalı iOS CI'de **12 native RunnerTests ve 2 simülatör akışı** geçti. **1.0.0 (1791142457)** Apple tarafından işlendi ve mevcut **Taskfulight Owner** grubuna eklendi; dış beta/App Store yayını yapılmadı. IPA bundle ID `com.aytzey.taskfulight`, kaynak provenance ve SHA-256 (`83cd4e7eb9ed93d9399214c68d65b7a7c5230e62b9e930b1d704a95bcdc2d5e1`) CI artefaktlarıyla eşleşiyor. Yeni sürümün fiziksel iPhone akışı bu çalışmada çalıştırılmadı.

TRIBE, [resmî proje açıklamasındaki](https://github.com/facebookresearch/tribev2) ortalama-denek fMRI yanıtını tahmin eder. Bu kullanıcının flörtünden ölçülmüş fMRI veya ikna olma olasılığı değildir; türetilmiş nöral eksen/puanlar bireysel davranışa doğrulanmış kalibrasyon sayılmaz.

Son dağıtım yalnız `tribe` servisini yeniledi. Diğer **165 konteynerin** kimliği, başlangıç zamanı, restart sayısı ve sağlık durumu değişmedi. Kaynak sınırları, auth/model mount'ları, Traefik ve native PyTorch CUDA korunuyor. Son sürüm öncesine hedefli dönüş `rollback-20261004T194256Z/rollback.sh`; bu aday seçimi çalışmasının tamamından önceki imaja dönüş `rollback-20261004T192626Z/rollback.sh`. Sunucudaki kanıt dizini `audit-tribe-search-final-20261004/`; kullanıcıya ait örnek metinler sadece yetkisi sınırlı canlı kabul kaydında tutulur ve Git/PR açıklamasına yazılmaz.

Son public HTTPS kabulünde kullanıcı girdisine karşı **4 gerçek, mock olmayan TRIBE ölçümü** (20.484 voxel) doğrulandı. Nöral türetilmiş puan orijinalde **32,357**, seçilende **38,637**; ayrı semantik bağlam puanı **42,5 → 60,5**, rapor puanı **25 → 40** oldu. Bunlar aynı kişinin gerçek davranışını ölçmez; yalnız bu işlemdeki model değerlendirmeleridir. Seçilen metin kısa ve verilen bilgilere sadık kaldı; doğal dil kalitesi hâlâ model değerlendirmesidir, evrensel ikna iddiası yoktur.

| Son canlı kabul ölçümü | Sonuç |
|---|---:|
| Tam yükleme | 4,783 sn |
| Orijinal skor + rapor | 24,807 sn |
| Üç aday üretimi + dört ölçüm + eleştirmen/seçim | 74,633 sn |
| Ayrı yeniden skor + rapor | 17,715 sn |
| Yeni adayların tekil native TRIBE hesabı | 0,196–0,240 sn |
| Health en uzun | 0,340 sn |
| 150 ms aralıklarla örneklenen cgroup `memory.current` tepe | 1,617 GiB |
| Örnek CUDA peak allocated | 6,695 GiB |
| Unload | 0,607 sn |

Bu kısa metin denemesinde dört TRIBE ölçümü yaklaşık bir saniyedir; toplam refine süresinin çoğu dış LLM beklemesidir. **74,633 saniye hızlı bir son kullanıcı yanıtı sayılmaz**; seçili ucuz sağlayıcının gecikmesi değişkendir. Yerelde aynı modelle aday üretimi yaklaşık beş saniye sürerken bu sunucu çağrısında belirgin gecikti. Bu örnekleme mutlak süreç RSS veya RAM tepe ölçümü değildir. En büyük metinler için RAM veya süre garantisi çıkarılmaz. Sıfır aktif GPU işi, başarılı unload, çağıran token'ın logout sonrası `401` alması ve diğer 165 servisin son kabul sonrasında da değişmemesi doğrulandı.

## Deneysel yapısal varsayım v2 — kaynak değişikliği

Kullanıcının talebiyle mevcut Hermes encoder korunur ve ölçülen model yanıt geometrisi varsayım olarak kullanılmaya devam eder. TRIBE checkpoint'inin beklediği encoder `meta-llama/Llama-3.2-3B`, mevcut encoder `NousResearch/Hermes-3-Llama-3.2-3B` olduğu için fizyolojik genelleme doğrulanmış sayılmaz. Bu uyumsuzluk sayısal gözlemleri gizleyen bir veto değildir; API, native/Python yollarında gerçek ve beklenen encoder kimliklerini açıkça taşır. Mutlak büyüklük, anatomik ROI, duygu veya ikna olasılığı olarak yorumlanmaz.

Her iz kendi ortalamasına bölünür. İlk/son çeyrek ve yarım pencerelerde giriş/kapanış oranları ile gruplanmış ardışık düşüşler gözlenir. Orijinalin en büyük yapısal açığı giriş, süreklilik veya kapanış deneyini belirler; aynı hedef Jev'in yazım öncesi kararına, yaklaşık kelime sırası hedeflerine ve bütün aday karşılaştırmalarına gider. Sayısal ağırlık mevcut çıktı-sağlığı ağırlığı × `min(1, (segment−1)/4)` × `min(1, iz_aralığı/0.0005)` olarak hesaplanır; son çarpan dört ondalıkta yuvarlanan izde küçük farkları azaltır. Aday etkisi aynı hedefin pencere değişimlerinin ortalamasıdır; her değişim −1…1 aralığında sınırlandırılır. Karşılaştırma ağırlığı orijinal/aday sayısal ağırlıklarının küçüğü × orijinal hedefin pencere desteği × yön uyumu × `min(1, ortak_pencere_sayısı/2)` olur. Farklı uzunluklar oransal pencerelerle karşılaştırılır; min/max değişim ve yön uyuşmazlığı duyarlılık olarak gösterilir, istatistiksel güven aralığı değildir. Ham uzamsal özetler ve farkları ayrı gözlemlerdir.

Doğruluk, asıl amaç, alıcıya saygı ve üslup kontrolleri önce uygulanır. En iyi bağlam puanının üç puan yakınındaki uygun adaylarda `seçim = clamp(bağlam + 3 × karşılaştırma_ağırlığı × etki, 0, 100)` kullanılır; gösterilen ve sıralanan puan aynıdır. Sıfır/sabit/mock çıktının ölçülebilir zamansal etkisi yoktur; zayıf ama değişken gerçek çıktı küçük, görünür katkı sağlar. Jev'in serbest kazanan tercihi bu politikayı geçersiz kılamaz. Opt-in akış tek `google/gemini-3.5-flash-lite` isteğinde üç aday üretir; düşük gizli reasoning ve toplam 1.536 token sınırı vardır. Yeniden yazım döngüsü, pahalı model veya sağlayıcı geri dönüşü yoktur. Bu bölüm kaynak politikasını açıklar; yeni canlı kabul sonuçları ayrıca kaydedilir.
